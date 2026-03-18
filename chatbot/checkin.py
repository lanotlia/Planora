from groq import Groq
import os
import json
from dotenv import load_dotenv
from chatbot.quiz import generate_questions, evaluate_answer, generate_flashcards_for_session, _get_base_interval
from chatbot.updater import extract_session_signals, generate_session_summary

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class CheckInSession:
    """
    Manages the full state of a single check-in conversation session.
    One instance is created per user per study session check-in.

    Flow:
        1. start()                      — opens the conversation
        2. process_checkin_response()   — handles how-did-it-go response
        3. process_material_response()  — handles study material or skip
        4. process_quiz_answer()        — handles each quiz answer
        5. get_session_data()           — returns all data for saving to Supabase
    """

    def __init__(self, user_profile: dict, subject_name: str, material: str = None):
        self.user_profile    = user_profile
        self.subject_name    = subject_name
        self.material        = material
        self.stage           = "checkin"
        self.checkin_text    = ""
        self.checkin_signals = {}
        self.questions       = []
        self.current_q_idx   = 0
        self.quiz_results    = []
        self.chat_history    = []

    def _call_llm(self, prompt: str) -> str:
        """Single helper for all LLM calls — keeps the code DRY"""
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

    def _build_system_context(self) -> str:
        profile      = self.user_profile
        user_name    = profile.get("user_name", "there")
        category     = profile.get("user_category", "uni_student")
        has_adhd     = profile.get("has_adhd", 0)
        has_dyslexia = profile.get("has_dyslexia", 0)
        has_autism   = profile.get("has_autism", 0)
        has_anxiety  = profile.get("has_anxiety", 0)

        base = f"""
        You are a warm, knowledgeable study coach doing a check-in with a student.
        Student name: {user_name}
        User category: {category}
        Subject just studied: {self.subject_name}

        Core communication rules:
        - Keep responses concise. Maximum 3 sentences unless giving feedback.
        - Never use bullet points or numbered lists.
        - Do not use generic phrases like "Great job!" or "Well done!"
        - Write in plain conversational English.
        - Never break character or refer to yourself as an AI.
        """

        condition_notes = []

        if has_adhd:
            condition_notes.append("""
            ADHD accommodations:
            - Use very short sentences. Maximum 15 words per sentence.
            - One idea per message only.
            - Be direct. Get to the point immediately.
            - Positive and energetic tone without being over the top.
            """)

        if has_dyslexia:
            condition_notes.append("""
            Dyslexia accommodations:
            - Maximum 2 sentences per response. Hard limit.
            - Use only simple, common words.
            - Do not draw attention to spelling errors. Ever.
            - Never use colons, semicolons, or complex punctuation.
            """)

        if has_autism:
            condition_notes.append("""
            Autism accommodations:
            - Be completely literal and explicit. No idioms or metaphors.
            - Always tell the student exactly what happens next.
            - Announce every stage transition clearly.
            - Give direct feedback: "That is correct." not "Nice one!"
            - Never use sarcasm.
            """)

        if has_anxiety:
            condition_notes.append("""
            Anxiety accommodations:
            - Frame everything as low-stakes and exploratory.
            - Never use: test, exam, correct answer, wrong, fail, score.
            - If they get something wrong, normalise it before explaining.
            - Emphasise effort over results.
            - Never mention time pressure.
            """)

        if condition_notes:
            base += "\n\nSpecific accommodations:\n" + "\n".join(condition_notes)

        return base

    def start(self) -> str:
        system_context = self._build_system_context()
        has_autism     = self.user_profile.get("has_autism", 0)
        has_anxiety    = self.user_profile.get("has_anxiety", 0)
        user_category  = self.user_profile.get("user_category", "uni_student")

        if user_category == "language_learner":
            subject_context = f"your {self.subject_name} practice session"
        elif user_category == "cert_candidate":
            subject_context = f"your {self.subject_name} exam prep session"
        else:
            subject_context = f"your {self.subject_name} session"

        if has_autism:
            n_questions = 4 if self.user_profile.get("has_adhd") else 5
            material_note = (
                f"After that, I will ask you {n_questions} questions about "
                f"what you studied in {self.subject_name}. "
                f"Then I will give you a summary of your session."
                if self.material else
                "After that, I will give you a summary of your session."
            )
            opening_prompt = f"""
            {system_context}
            Greet the student by name.
            Ask one direct question: how did {subject_context} go?
            Then explain the session structure: "{material_note}"
            Three sentences maximum. Be direct. No idioms.
            """

        elif has_anxiety:
            opening_prompt = f"""
            {system_context}
            Start the check-in warmly. Ask how {subject_context} went
            as a casual conversation, not an evaluation.
            Keep it to 2 sentences. Make it feel safe and unhurried.
            """

        else:
            opening_prompt = f"""
            {system_context}
            Start the check-in. Ask how {subject_context} went.
            Ask about focus and energy naturally. 2-3 sentences maximum.
            """

        return self._call_llm(opening_prompt)

    def process_checkin_response(self, user_message: str) -> str:
        self.checkin_text    = user_message
        self.checkin_signals = extract_session_signals(
            user_message,
            self.user_profile.get("user_category", "uni_student")
        )

        system_context = self._build_system_context()

        if self.material:
            return self._transition_to_quiz(system_context)

        self.stage  = "material_prompt"
        has_autism  = self.user_profile.get("has_autism", 0)
        has_anxiety = self.user_profile.get("has_anxiety", 0)

        if has_autism:
            return (
                f"You can now share your study material for {self.subject_name}. "
                f"Paste your notes or type 'skip' to continue without material."
            )

        elif has_anxiety:
            prompt = f"""
            {system_context}
            The student said: "{user_message}"
            Acknowledge warmly in one sentence. Focus on effort not outcome.
            Then gently ask if they want to share what they covered today.
            Make it feel optional. Tell them they can type 'skip'.
            Never use the word 'test', 'quiz', or 'upload'.
            Two sentences total maximum.
            """
        else:
            prompt = f"""
            {system_context}
            The student said: "{user_message}"
            Acknowledge in one sentence.
            Then ask them to share their notes, slides or any material
            so you can quiz them and create flashcards.
            Make it feel optional. Say they can type 'skip'.
            Two sentences total maximum.
            """

        return self._call_llm(prompt)

    def process_material_response(self, user_message: str) -> str:
        system_context = self._build_system_context()
        skip_phrases   = ["skip", "no", "nope", "nothing", "i don't have",
                          "i dont have", "no material", "no notes", "pass"]

        skipped = any(phrase in user_message.lower().strip() for phrase in skip_phrases)

        if skipped or len(user_message.strip()) < 10:
            self.stage = "summary"
            close_prompt = f"""
            {system_context}
            The student chose not to share material for {self.subject_name}.
            Acknowledge that warmly in one sentence.
            Give one practical tip for their next session. 2 sentences total.
            """
            return self._call_llm(close_prompt)

        else:
            self.material = user_message
            return self._transition_to_quiz(system_context)

    def _transition_to_quiz(self, system_context: str) -> str:
        self.questions = generate_questions(
            material     = self.material,
            user_profile = self.user_profile,
            subject_name = self.subject_name,
        )
        self.stage = "quiz"

        has_autism = self.user_profile.get("has_autism", 0)

        if has_autism:
            transition_instruction = f"""
            {system_context}
            Acknowledge you received the material in one literal sentence.
            Then state: "I am going to ask you {len(self.questions)} 
            questions about {self.subject_name} now."
            """
        else:
            transition_instruction = f"""
            {system_context}
            Acknowledge you have their material in one sentence.
            Then naturally say you want to see what stuck from the session.
            Two sentences total maximum.
            """

        response_text = self._call_llm(transition_instruction)
        first_q       = self._format_question(self.questions[0])
        return f"{response_text}\n\n{first_q}"

    def _format_question(self, question: dict) -> str:
        text = f"Question {question['id']}: {question['question']}"
        if question.get("options"):
            text += "\n\n" + "\n".join(question["options"])
        return text

    def process_quiz_answer(self, user_answer: str):
        current_question = self.questions[self.current_q_idx]

        result = evaluate_answer(
            question       = current_question["question"],
            correct_answer = current_question["correct_answer"],
            user_answer    = user_answer,
            question_type  = current_question["type"],
            user_profile   = self.user_profile
        )

        result["question_id"] = current_question["id"]
        self.quiz_results.append(result)
        self.current_q_idx += 1

        if self.current_q_idx < len(self.questions):
            feedback = result["feedback"]
            next_q   = self._format_question(self.questions[self.current_q_idx])
            return f"{feedback}\n\n{next_q}"
        else:
            self.stage = "summary"
            return self._generate_final_summary()

    def _generate_final_summary(self) -> tuple:
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

        # ── Auto-generate flashcards ──────────────────────────────────────────
        flashcards = []
        if self.material:
            flashcards = generate_flashcards_for_session(
                material      = self.material,
                user_profile  = self.user_profile,
                subject_name  = self.subject_name,
                quiz_results  = self.quiz_results,
                base_interval = _get_base_interval(score_pct)
            )
            summary["flashcards"]      = flashcards
            summary["flashcard_count"] = len(flashcards)
        else:
            summary["flashcards"]      = []
            summary["flashcard_count"] = 0

        # ── Performance note ──────────────────────────────────────────────────
        if has_autism:
            performance_note = (
                f"Session complete. Score: {score_pct}%. "
                f"Correct answers: {summary['correct_answers']} out of {summary['total_questions']}. "
                f"Understanding level: {understanding}."
            )
        elif has_anxiety:
            if understanding == "strong":
                performance_note = f"You clearly retained a lot from today's session on {self.subject_name}."
            elif understanding == "moderate":
                performance_note = f"You got some solid things from today's session. A few areas are worth revisiting when you feel ready."
            else:
                performance_note = f"This material is still settling in — completely normal for new content."
        else:
            if understanding == "strong":
                performance_note = f"You scored {score_pct}% — solid retention of {self.subject_name}."
            elif understanding == "moderate":
                performance_note = f"You scored {score_pct}%. Good progress, with a few areas worth revisiting."
            else:
                performance_note = f"You scored {score_pct}%. This material needs another pass — completely expected at this stage."

        # ── Flashcard note ────────────────────────────────────────────────────
        flashcard_note = ""
        if flashcards:
            due_date = flashcards[0].get("next_review_date", next_review)
            flashcard_note = (
                f" I have created {len(flashcards)} flashcards for you to review "
                f"before your next session on {due_date}."
            )

        summary_message = (
            f"{performance_note}{flashcard_note} "
            f"Next review of {self.subject_name} scheduled for {next_review}.\n\n"
            f"{encouragement}"
        )

        return summary_message, summary

    def get_session_data(self) -> dict:
        return {
            "user_profile":    self.user_profile,
            "subject_name":    self.subject_name,
            "checkin_text":    self.checkin_text,
            "checkin_signals": self.checkin_signals,
            "quiz_results":    self.quiz_results,
            "stage":           self.stage,
        }