from google import genai
import os
from dotenv import load_dotenv
from chatbot.quiz import generate_questions, evaluate_answer
from chatbot.updater import extract_session_signals, generate_session_summary

load_dotenv()

# Configure once at module level — never inside a function or class
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def get_client():
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class CheckInSession:
    """
    Manages the full state of a single check-in conversation session.
    One instance is created per user per study session check-in.

    Flow:
        1. start()                      — opens the conversation
        2. process_checkin_response()   — handles how-did-it-go response,
                                          extracts signals, transitions to quiz
        3. process_quiz_answer()        — handles each quiz answer, gives feedback,
                                          generates final summary when done
        4. get_session_data()           — returns all data for saving to Supabase
    """

    def __init__(
        self,
        user_profile: dict,
        subject_name: str,
        material: str = None
    ):
        self.user_profile    = user_profile
        self.subject_name    = subject_name
        self.material        = material     # pasted text, extracted PDF, or None
        self.stage           = "checkin"    # checkin → quiz → summary → complete
        self.checkin_text    = ""
        self.checkin_signals = {}
        self.questions       = []
        self.current_q_idx   = 0
        self.quiz_results    = []
        self.chat_history    = []

        # Single Gemini chat session — maintains context across the conversation
    prompt_text = f"""..."""
    response = client.models.generate_content(
     model="gemini-1.5-flash",
     contents=prompt_text
)

    # ─────────────────────────────────────────────────────────────────────────
    # SYSTEM CONTEXT
    # Single source of truth for how Gemini communicates with this user.
    # Called at the start of every prompt sent to Gemini.
    # ─────────────────────────────────────────────────────────────────────────

    def _build_system_context(self) -> str:
        """
        Build the system context string injected into every Gemini prompt.
        Layers general communication rules with condition-specific
        accommodations for ADHD, dyslexia, autism, and anxiety.
        """

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

        Core communication rules — always apply these no matter what:
        - Keep responses concise. Maximum 3 sentences unless giving feedback.
        - Never use bullet points or numbered lists in your responses.
        - Do not use generic phrases like "Great job!" or "Well done!"
          Be specific about what they actually did.
        - Write in plain conversational English.
        - Never break character or refer to yourself as an AI.
        """

        condition_notes = []

        # ── ADHD accommodations ───────────────────────────────────────────────
        if has_adhd:
            condition_notes.append("""
            ADHD accommodations — apply these strictly:
            - Use very short sentences. Maximum 15 words per sentence.
            - One idea per message only. Never combine multiple points in one message.
            - Be direct. Get to the point in the very first sentence.
            - Positive and energetic tone without being over the top.
            - No long reflective paragraphs. Keep it moving.
            """)

        # ── Dyslexia accommodations ───────────────────────────────────────────
        if has_dyslexia:
            condition_notes.append("""
            Dyslexia accommodations — apply these strictly:
            - Maximum 2 sentences per response. This is a hard limit.
            - Use only simple, common words. Nothing complex or multisyllabic
              unless it is the subject term itself.
            - Never write a paragraph longer than 2 lines.
            - Put each new idea on its own line with a line break between them.
            - When giving feedback, structure it as:
              Line 1: Was the answer right or not, stated simply and directly.
              Line 2: The one thing to remember. Nothing else.
            - Do not draw attention to spelling errors in the student's answers.
              Ever. Ignore all spelling and focus only on meaning.
            - Never use colons, semicolons, or complex punctuation.
            """)

        # ── Autism accommodations ─────────────────────────────────────────────
        if has_autism:
            condition_notes.append("""
            Autism accommodations — apply these strictly:
            - Be completely literal and explicit at all times.
              No idioms, metaphors, sarcasm, or vague language of any kind.
            - Always tell the student exactly what is about to happen before
              it happens. Unexpected transitions cause distress.
              Example: "I am going to ask you 4 questions now. Take your time."
            - Announce every stage transition clearly and explicitly.
            - Maintain a predictable, consistent tone throughout.
              No sudden shifts in energy, warmth, or style.
            - Give direct feedback using clear factual statements.
              Say: "That is correct." not "Ooh nice one!"
              Say: "That is not correct." not "Not quite, but good try!"
            - Never use sarcasm even if lighthearted.
            - Never use phrases like "let's dive in", "sounds like",
              "that's a tough one", or any idiom.
            """)

        # ── Anxiety accommodations ────────────────────────────────────────────
        if has_anxiety:
            condition_notes.append("""
            Anxiety accommodations — apply these strictly:
            - Frame everything as low-stakes and exploratory, not evaluative.
            - Never use the words: test, exam, correct answer, wrong, fail,
              score, performance, or grade in your responses.
            - Instead of "Let me test you" say "Let's see what stuck."
            - If they get something wrong, normalise it immediately before
              explaining. Example: "That's a common mix-up — the key thing
              to remember is [correction]."
            - Always acknowledge what they understood correctly even in a
              wrong answer, before explaining what to correct.
            - Emphasise effort and the process of learning over results.
            - Never mention time pressure or urgency.
            - Keep tone warm and unhurried throughout.
            """)

        if condition_notes:
            base += "\n\nSpecific accommodations for this student:\n"
            base += "\n".join(condition_notes)

        return base

    # ─────────────────────────────────────────────────────────────────────────
    # SESSION STAGES
    # ─────────────────────────────────────────────────────────────────────────

    def start(self) -> str:
        """
        Generate the opening check-in message.
        Autism users receive an explicit preview of the session structure
        upfront so there are no unexpected transitions later.
        """

        system_context = self._build_system_context()
        has_autism     = self.user_profile.get("has_autism", 0)
        has_anxiety    = self.user_profile.get("has_anxiety", 0)
        user_category  = self.user_profile.get("user_category", "uni_student")

        # Build subject context phrase based on category
        if user_category == "language_learner":
            subject_context = f"your {self.subject_name} practice session"
        elif user_category == "cert_candidate":
            subject_context = f"your {self.subject_name} exam prep session"
        else:
            subject_context = f"your {self.subject_name} session"

        # Autism users need the full session structure explained upfront
        # This prevents distress from unexpected transitions mid-conversation
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
            Then on a new line, explain the session structure clearly:
            "{material_note}"

            Three sentences maximum total. Be direct. No idioms.
            """

        # Anxiety users need the tone set as low-stakes immediately
        elif has_anxiety:
            opening_prompt = f"""
            {system_context}

            Start the check-in warmly. Ask how {subject_context} went
            in a way that feels like a casual conversation, not an evaluation.
            Ask about their energy and focus naturally.
            Keep it to 2 sentences. Make it feel safe and unhurried.
            """

        else:
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
        Process the user's how-did-it-go response.
        Extracts session signals via Gemini, generates quiz questions
        from the study material if provided, and transitions to the quiz.
        If no material was provided, closes the session with a practical tip.
        """

        self.checkin_text    = user_message
        self.checkin_signals = extract_session_signals(
            user_message,
            self.user_profile.get("user_category", "uni_student")
        )

        system_context = self._build_system_context()
        has_autism     = self.user_profile.get("has_autism", 0)
        has_anxiety    = self.user_profile.get("has_anxiety", 0)

        if self.material:
            # Generate questions from study material
            self.questions = generate_questions(
                material     = self.material,
                user_profile = self.user_profile,
                subject_name = self.subject_name
            )
            self.stage = "quiz"

            # Build the transition message to the quiz
            # Each condition gets a different transition style
            if has_autism:
                # Autism: explicit, literal, no idioms, state exactly what happens
                transition_prompt = f"""
                {system_context}

                The student said: "{user_message}"

                Write exactly 2 sentences:
                Sentence 1: Acknowledge what they said literally and factually.
                  No idioms. No phrases like "sounds like" or "tough one."
                  Good: "You completed the session and found parts of it difficult."
                  Bad: "Sounds like it was a tough one!"
                Sentence 2: State explicitly what happens now.
                  Good: "I am going to ask you {len(self.questions)} questions
                  about {self.subject_name} now."
                  Bad: "Let's see what stuck!"

                No other sentences. No filler. No encouragement phrases.
                """

            elif has_anxiety:
                # Anxiety: low-stakes framing, no test language
                transition_prompt = f"""
                {system_context}

                The student said: "{user_message}"

                Acknowledge their response warmly in one sentence.
                Then in one sentence, transition to the questions using
                low-stakes language. Say something like "let's see what
                stuck from today" — never "let me test you" or "quiz time."
                Two sentences total maximum.
                """

            else:
                transition_prompt = f"""
                {system_context}

                The student said: "{user_message}"

                Acknowledge their response briefly in one sentence.
                Then naturally say you want to go through what they
                just studied together. Two sentences total maximum.
                """

            transition   = self.chat.send_message(transition_prompt)
            first_q      = self._format_question(self.questions[0])
            return f"{transition.text.strip()}\n\n{first_q}"

        else:
            # No material provided — close with a practical tip
            self.stage = "summary"

            close_prompt = f"""
            {system_context}

            The student said: "{user_message}"

            Acknowledge what they shared.
            Give one practical, specific tip for their next session based
            on what they told you. Keep it to 2 sentences maximum.
            """

            close = self.chat.send_message(close_prompt)
            return close.text.strip()

    def process_quiz_answer(self, user_answer: str):
        """
        Evaluate the current question answer.
        Returns a string response if more questions remain.
        Returns a tuple (message, summary_dict) when the quiz is complete.
        """

        current_question = self.questions[self.current_q_idx]

        # Evaluate the answer using Gemini
        result = evaluate_answer(
            question      = current_question["question"],
            correct_answer= current_question["correct_answer"],
            user_answer   = user_answer,
            question_type = current_question["type"],
            user_profile  = self.user_profile
        )

        result["question_id"] = current_question["id"]
        self.quiz_results.append(result)
        self.current_q_idx += 1

        # More questions remain — give feedback and ask the next one
        if self.current_q_idx < len(self.questions):
            feedback = result["feedback"]
            next_q   = self._format_question(self.questions[self.current_q_idx])
            return f"{feedback}\n\n{next_q}"

        # All questions answered — generate the final summary
        else:
            self.stage = "complete"
            return self._generate_final_summary()

    # ─────────────────────────────────────────────────────────────────────────
    # HELPER METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def _format_question(self, question: dict) -> str:
        """
        Format a question object into a readable string for the user.
        Multiple choice questions get their options displayed.
        """

        text = f"**Question {question['id']}:** {question['question']}"

        if question.get("options"):
            options_text = "\n".join(question["options"])
            text += f"\n\n{options_text}"

        return text

    def _generate_final_summary(self) -> tuple:
        """
        Generate the closing summary message and summary data object.
        The message is adapted for the user's learning profile.
        Returns a tuple: (message_string, summary_dict)
        """

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

        # ── Autism: explicit, factual, literal performance summary ────────────
        if has_autism:
            performance_note = (
                f"Session complete. "
                f"Score: {score_pct}%. "
                f"Correct answers: {summary['correct_answers']} "
                f"out of {summary['total_questions']}. "
                f"Understanding level: {understanding}."
            )

        # ── Anxiety: remove score from the framing entirely ───────────────────
        elif has_anxiety:
            if understanding == "strong":
                performance_note = (
                    f"You clearly retained a lot from today's session on "
                    f"{self.subject_name}."
                )
            elif understanding == "moderate":
                performance_note = (
                    f"You got some solid things from today's session. "
                    f"A few areas are worth revisiting when you feel ready."
                )
            else:
                performance_note = (
                    f"This material is still settling in — that's completely "
                    f"normal for new content and it means today's session was "
                    f"still useful."
                )

        # ── Everyone else: standard performance note with score ───────────────
        else:
            if understanding == "strong":
                performance_note = (
                    f"You scored {score_pct}% — solid retention of "
                    f"{self.subject_name}."
                )
            elif understanding == "moderate":
                performance_note = (
                    f"You scored {score_pct}%. Good progress, with a few "
                    f"areas worth revisiting."
                )
            else:
                performance_note = (
                    f"You scored {score_pct}%. This material needs another "
                    f"pass — completely expected at this stage."
                )

        summary_message = (
            f"{performance_note} "
            f"Next review of {self.subject_name} scheduled for {next_review}."
            f"\n\n{encouragement}"
        )

        return summary_message, summary

    def get_session_data(self) -> dict:
        """
        Return all session data for saving to Supabase.
        Called after the session is complete.
        """

        return {
            "user_profile":    self.user_profile,
            "subject_name":    self.subject_name,
            "checkin_text":    self.checkin_text,
            "checkin_signals": self.checkin_signals,
            "quiz_results":    self.quiz_results,
            "stage":           self.stage,
        }