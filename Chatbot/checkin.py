import google.generativeai as genai
import os
import json
from chatbot.quiz import generate_questions, evaluate_answer
from chatbot.updater import extract_session_signals, generate_session_summary

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel("gemini-1.5-flash")


class CheckInSession:
    """
    Manages the state of a single check-in conversation.
    One instance per user per session.
    """

    def __init__(self, user_profile: dict, subject_name: str, material: str = None):
        self.user_profile   = user_profile
        self.subject_name   = subject_name
        self.material       = material          # PDF text or pasted notes
        self.stage          = "checkin"         # checkin → quiz → summary
        self.checkin_text   = ""
        self.checkin_signals= {}
        self.questions      = []
        self.current_q_idx  = 0
        self.quiz_results   = []
        self.chat_history   = []                # for Gemini multi-turn

        # Start the Gemini chat session
        self.chat = gemini.start_chat(history=[])

    def start(self) -> str:
        """Generate the opening check-in message"""

        user_name    = self.user_profile.get("user_name", "there")
        category     = self.user_profile.get("user_category", "uni_student")
        has_adhd     = self.user_profile.get("has_adhd", 0)

        # Personalise opening based on category
        if category == "language_learner":
            subject_context = f"your {self.subject_name} practice session"
        elif category == "cert_candidate":
            subject_context = f"your {self.subject_name} exam prep session"
        else:
            subject_context = f"your {self.subject_name} session"

        system_context = f"""
        You are a warm, encouraging study coach doing a brief check-in with a student.
        Student name: {user_name}
        User category: {category}
        Has ADHD: {bool(has_adhd)}
        Subject just studied: {self.subject_name}
        
        Keep responses conversational and brief.
        {'Use short sentences — this student has ADHD.' if has_adhd else ''}
        Do not use bullet points in your responses.
        Be warm but not over the top.
        """

        opening_prompt = f"""
        {system_context}
        
        Start the check-in. Ask how {subject_context} went.
        Ask about their focus and energy in a natural conversational way.
        Keep it to 2-3 sentences maximum.
        """

        response = self.chat.send_message(opening_prompt)
        return response.text.strip()

    def process_checkin_response(self, user_message: str) -> str:
        """
        Process the user's check-in response, extract signals,
        then transition to quiz if material is available.
        """

        self.checkin_text    = user_message
        self.checkin_signals = extract_session_signals(
            user_message,
            self.user_profile.get("user_category", "uni_student")
        )

        # Generate questions if material was provided
        if self.material:
            self.questions = generate_questions(
                material      = self.material,
                user_profile  = self.user_profile,
                subject_name  = self.subject_name,
                struggle      = self.user_profile.get("struggle", "understanding"),
                has_adhd      = bool(self.user_profile.get("has_adhd", 0)),
                current_level = self.user_profile.get("current_level"),
                n_questions   = 4 if self.user_profile.get("has_adhd") else 5
            )
            self.stage = "quiz"

            # Transition message into quiz
            transition = self.chat.send_message(
                f"""
                The student said: "{user_message}"
                
                Acknowledge their response briefly in one sentence.
                Then naturally transition into saying you want to test them 
                on what they just studied.
                Keep it friendly and brief — 2 sentences total.
                Do not list the questions yet.
                """
            )

            # Return transition + first question
            first_question = self._format_question(self.questions[0])
            return f"{transition.text.strip()}\n\n{first_question}"

        else:
            # No material provided — just do emotional check-in and close
            self.stage = "summary"
            close = self.chat.send_message(
                f"""
                The student said: "{user_message}"
                Acknowledge warmly, give one practical tip for their next session
                based on what they shared. Keep it to 2 sentences.
                """
            )
            return close.text.strip()

    def process_quiz_answer(self, user_answer: str) -> str:
        """
        Evaluate the current question answer and either ask the next
        question or generate the session summary.
        """

        current_question = self.questions[self.current_q_idx]

        # Evaluate answer
        result = evaluate_answer(
            question      = current_question["question"],
            correct_answer= current_question["correct_answer"],
            user_answer   = user_answer,
            question_type = current_question["type"],
            user_category = self.user_profile.get("user_category", "uni_student")
        )

        result["question_id"] = current_question["id"]
        self.quiz_results.append(result)

        self.current_q_idx += 1

        # More questions remaining
        if self.current_q_idx < len(self.questions):
            feedback    = result["feedback"]
            next_q      = self._format_question(self.questions[self.current_q_idx])
            return f"{feedback}\n\n{next_q}"

        # Quiz complete — generate summary
        else:
            self.stage = "summary"
            return self._generate_final_summary()

    def _format_question(self, question: dict) -> str:
        """Format a question object into a readable string"""

        text = f"**Question {question['id']}:** {question['question']}"

        # Add options for multiple choice
        if question.get("options"):
            options_text = "\n".join(question["options"])
            text += f"\n\n{options_text}"

        return text

    def _generate_final_summary(self) -> str:
        summary = generate_session_summary(
            checkin_signals = self.checkin_signals,
            quiz_results    = self.quiz_results,
            user_profile    = self.user_profile,
            subject_name    = self.subject_name
        )

        score_pct     = summary["quiz_score"]
        next_review   = summary["next_review_date"]
        encouragement = summary["encouragement"]
        understanding = summary["understanding"]
        has_anxiety   = self.user_profile.get("has_anxiety", 0)
        has_autism    = self.user_profile.get("has_autism", 0)

        # Anxiety users should not see a raw percentage score prominently
        if has_anxiety:
            if understanding == "strong":
                performance_note = f"You clearly retained a lot from {self.subject_name} today."
            elif understanding == "moderate":
                performance_note = f"You got some good things from today's session. A few areas are worth revisiting."
            else:
                performance_note = f"This material needs another pass — that's completely normal for new content."

        # Autism users get explicit, literal summary
        elif has_autism:
            performance_note = (
                f"Your score: {score_pct}%. "
                f"Questions correct: {summary['correct_answers']} out of {summary['total_questions']}. "
                f"Understanding level: {understanding}."
            )

        # Everyone else gets the standard message
        else:
            if understanding == "strong":
                performance_note = f"You scored {score_pct}% — solid retention of {self.subject_name}."
            elif understanding == "moderate":
                performance_note = f"You scored {score_pct}%. Good progress, a few gaps worth revisiting."
            else:
                performance_note = f"You scored {score_pct}%. This material needs another pass — completely fine at this stage."

        summary_message = (
            f"{performance_note} "
            f"Next review of {self.subject_name} scheduled for {next_review}.\n\n"
            f"{encouragement}"
        )

        return summary_message, summary 

def get_session_data(self) -> dict:
        """Return all session data for saving to Supabase"""
        return {
            "user_profile":      self.user_profile,
            "subject_name":      self.subject_name,
            "checkin_text":      self.checkin_text,
            "checkin_signals":   self.checkin_signals,
            "quiz_results":      self.quiz_results,
            "stage":             self.stage,
        }