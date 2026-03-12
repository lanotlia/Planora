import google.generativeai as genai
import os
import json
from dotenv import load_dotenv
from typer import prompt

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def extract_text_from_material(material: str) -> str:
    """Clean and truncate material to fit context window"""
    return material[:8000] if len(material) > 8000 else material


def generate_questions(
    material: str,
    user_profile: dict,
    subject_name: str,
    n_questions: int = 5
) -> list:
    """
    Generate questions tailored to user category and full learning profile.
    Accounts for ADHD, dyslexia, autism, and anxiety in both question
    style, format, and language used.
    """

    material      = extract_text_from_material(material)
    user_category = user_profile.get("user_category", "uni_student")
    struggle      = user_profile.get("struggle", "understanding")
    current_level = user_profile.get("current_level")
    has_adhd      = user_profile.get("has_adhd", 0)
    has_dyslexia  = user_profile.get("has_dyslexia", 0)
    has_autism    = user_profile.get("has_autism", 0)
    has_anxiety   = user_profile.get("has_anxiety", 0)

    # ── Adjust question count based on conditions ─────────────────────────────
    # Fewer questions reduce cognitive fatigue for these profiles
    if has_adhd or has_anxiety or has_dyslexia:
        n_questions = min(n_questions, 4)

    # ── Build category-specific question style ────────────────────────────────
    if user_category == "language_learner":
        question_style = f"""
        The student is a {current_level or "intermediate"} language learner
        studying {subject_name}.

        Generate a MIX of these question types:
        - Vocabulary: "What does [word from the material] mean?"
        - Translation: "Translate this sentence into English: [sentence]"
        - Fill in the blank: "[Sentence with a gap] — fill in the missing word"
        - Usage: "Use the word [word] in your own sentence"
        - Comprehension: A short question about the meaning of a passage

        Weight vocabulary and fill-in-the-blank questions most heavily.
        Use actual words and sentences directly from the material provided.
        """

    elif user_category == "cert_candidate":
        question_style = f"""
        Generate exam-style questions for a student preparing for {subject_name}.

        Use these question types:
        - Multiple choice with 4 options labelled A, B, C, D
        - Scenario-based: "In this situation, what would you do..."
        - Definition: "Define [term] in your own words"
        - Application: "How would you apply [concept] to [scenario]"

        Make questions specific, precise, and realistic to the exam format.
        Always include the correct answer in the question object.
        """

    elif user_category == "uni_student":
        question_style = f"""
        Generate university-level questions for {subject_name} that test
        genuine understanding, not surface recall.

        Use these question types:
        - Concept: "Explain [concept] in your own words"
        - Application: "How does [concept] apply to [scenario from material]"
        - Compare: "What is the difference between [x] and [y]"
        - Critical thinking: "Why does [phenomenon] occur?"

        Prioritise understanding and reasoning over memorisation.
        """

    elif user_category == "self_study":
        question_style = f"""
        Generate practical comprehension questions for someone self-studying
        {subject_name}.

        Use these question types:
        - "Explain [concept] as if teaching someone else"
        - "What is the main purpose of [concept/tool/idea]?"
        - "Give an example of how you would use [concept]"
        - "What would happen if [condition changed]?"

        Focus on practical understanding and real-world application.
        """

    else:
        question_style = f"""
        Generate 5 comprehension questions about {subject_name} based on the
        material. Mix recall, understanding, and application question types.
        """

    # ── AUTISM OVERRIDE — replaces category style entirely ───────────────────
    # Autism requires unambiguous, literal questions with one clear answer.
    # Open-ended questions cause confusion and distress — override completely.
    if has_autism:
        question_style = f"""
        This student needs completely explicit and unambiguous questions
        about {subject_name}.

        ONLY generate these question types:
        - Definition: "What is [term]?" — one clear factual answer
        - True or False: "True or false: [statement]"
        - Multiple choice with 4 options — one is clearly and objectively correct
        - Specific recall: "What year did [event] happen?" or
          "Name the [specific thing] described in the material"

        NEVER generate:
        - "Explain in your own words..."
        - "What do you think about..."
        - "How would you approach..."
        - "In your opinion..."
        - Any question that has more than one valid correct answer
        - Any hypothetical or open-ended scenario questions

        Every single question must have exactly one objectively correct answer.
        Be precise. Be literal. No ambiguity of any kind.
        """

    # ── ADHD modifications ────────────────────────────────────────────────────
    adhd_note = ""
    if has_adhd:
        adhd_note = """
        ADHD accommodations for question formatting:
        - Keep each question SHORT and direct. One sentence maximum.
        - No long preambles or scene-setting before the question.
        - One concept per question only. Never combine two ideas.
        - Questions should be immediately obvious what is being asked.
        - Avoid multi-part questions entirely.
        """

    # ── Dyslexia modifications ────────────────────────────────────────────────
    dyslexia_note = ""
    if has_dyslexia:
        dyslexia_note = """
        Dyslexia accommodations for question formatting:
        - Keep question text short. One sentence per question maximum.
        - Prefer multiple choice questions — they reduce writing demand.
        - Avoid questions that require long written answers.
        - Use simple, everyday words in the question text itself.
          Do not use complex vocabulary in how the question is phrased.
        - Never ask the student to copy, spell out, or transcribe text.
        - Accept answers that are conceptually correct even if spelling
          is imperfect — do not penalise spelling in evaluation.
        - Avoid questions that require reading long passages to answer.
        """

    # ── Anxiety modifications ─────────────────────────────────────────────────
    anxiety_note = ""
    if has_anxiety:
        anxiety_note = """
        Anxiety accommodations for question formatting:
        - Frame questions as curiosity not testing.
          Use: "What do you remember about..." not "What is the correct answer for..."
        - Avoid the words "test", "correct", "wrong", "fail", "must" in question text.
        - Make questions feel like a natural conversation, not an interrogation.
        - Prefer multiple choice where possible — less performance pressure
          than open-ended questions.
        - Questions should feel achievable, not intimidating.
        """

    # ── Struggle-based modifications ──────────────────────────────────────────
    struggle_note = ""
    if struggle == "understanding":
        struggle_note = "Weight questions toward conceptual explanation and reasoning over facts."
    elif struggle == "remembering":
        struggle_note = "Include more definition and active recall questions to strengthen memory."
    elif struggle in ["vocab_doesnt_stick", "grammar_confusion"]:
        struggle_note = "Focus heavily on vocabulary retention and correct usage in context."
    elif struggle == "cant_produce":
        struggle_note = "Include questions that require the student to produce language, not just recognise it."

    # ── Final prompt ──────────────────────────────────────────────────────────
    prompt = f"""
    You are a study coach reviewing material with a student who just
    finished studying. Your job is to generate questions that help
    them consolidate what they learned.

    Subject: {subject_name}

    {question_style}

    {adhd_note}

    {dyslexia_note}

    {anxiety_note}

    {struggle_note}

    Based on the following study material, generate exactly {n_questions}
    questions.

    CRITICAL: Return ONLY a valid JSON array. No extra text before or after.
    No markdown. No explanation. Just the JSON array.

    Required format:
    [
      {{
        "id": 1,
        "question": "The question text here",
        "type": "concept/vocabulary/multiple_choice/fill_blank/application/definition/true_false",
        "options": ["A. option", "B. option", "C. option", "D. option"],
        "correct_answer": "The correct answer here",
        "follow_up": "A short hint if they get it wrong — one sentence"
      }}
    ]

    Notes on the format:
    - "options" should only be populated for multiple_choice and true_false questions.
      Set it to null for all other types.
    - "correct_answer" must always be filled in — never leave it empty.
    - "follow_up" should never give away the answer, just point the student
      in the right direction.

    Study material:
    {material}
    """

    response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt
)
    raw = response.text.strip()

    # Strip markdown code blocks if Gemini adds them
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]

    questions = json.loads(raw.strip())
    return questions


def evaluate_answer(
    question: str,
    correct_answer: str,
    user_answer: str,
    question_type: str,
    user_profile: dict
) -> dict:
    """
    Evaluate a user's answer and return a score with feedback.
    Uses Gemini to handle natural language answers fairly.
    Feedback style is adapted for the user's learning profile.
    """

    has_dyslexia = user_profile.get("has_dyslexia", 0)
    has_autism   = user_profile.get("has_autism", 0)
    has_anxiety  = user_profile.get("has_anxiety", 0)
    has_adhd     = user_profile.get("has_adhd", 0)

    # ── Build condition-specific feedback instructions ────────────────────────
    feedback_style = ""

    if has_autism:
        feedback_style = """
        Feedback style for this student:
        - Be completely direct and literal. State clearly if the answer
          was correct or incorrect.
        - Use: "That is correct." or "That is not correct."
        - Then give exactly one factual sentence explaining what the
          right answer is if they were wrong.
        - No praise phrases. No "well done" or "nice try."
        - No emotive language. Factual only.
        - Maximum 2 sentences total.
        """

    elif has_dyslexia:
        feedback_style = """
        Feedback style for this student:
        - Use very simple, short words.
        - Maximum 2 short sentences.
        - Do not mention or draw attention to spelling errors.
          Evaluate the meaning of their answer only.
        - If correct: one short sentence confirming it.
        - If wrong: one short sentence with what to remember.
        - No complex vocabulary in the feedback itself.
        """

    elif has_adhd:
        feedback_style = """
        Feedback style for this student:
        - Be brief and direct. Maximum 2 sentences.
        - Get to the point immediately.
        - If correct: acknowledge it and move on.
        - If wrong: give the key correction in one short sentence.
        - Keep energy positive but not over the top.
        """

    elif has_anxiety:
        feedback_style = """
        Feedback style for this student:
        - Never use the words "wrong", "incorrect", "failed", or "mistake."
        - If they got it right: acknowledge specifically what they understood.
        - If they got it wrong: normalise it first, then explain.
          Example: "This one trips a lot of people up — the key thing to
          remember is [correction]."
        - Emphasise what they did understand, even in a wrong answer.
        - Maximum 2 sentences. Warm but not patronising.
        """

    else:
        feedback_style = """
        Feedback style:
        - Brief and encouraging. Maximum 2 sentences.
        - If correct: acknowledge it specifically.
        - If wrong: explain the correct answer clearly without being harsh.
        - End on a forward-looking note when possible.
        """

    # ── Dyslexia: ignore spelling in evaluation ───────────────────────────────
    spelling_note = ""
    if has_dyslexia:
        spelling_note = """
        IMPORTANT: Do not penalise spelling errors in the student's answer.
        Evaluate only whether the meaning and concept is correct.
        A misspelled answer that shows correct understanding should score
        the same as a perfectly spelled one.
        """

    prompt = f"""
    You are a fair and encouraging study coach evaluating a student's answer.

    Question: {question}
    Question type: {question_type}
    Correct answer: {correct_answer}
    Student's answer: {user_answer}

    {feedback_style}

    {spelling_note}

    Scoring guide:
    - 3: Fully correct, shows clear understanding
    - 2: Mostly correct, minor gaps or imprecision
    - 1: Partially correct, right direction but missing key points
    - 0: Incorrect or shows fundamental misunderstanding

    Be generous with partial credit for natural language answers.
    For multiple choice and true/false, only award 3 or 0.

    Respond ONLY with a valid JSON object. No extra text. No markdown.

    {{
      "score": 0-3,
      "correct": true/false,
      "partial": true/false,
      "feedback": "Your feedback here following the style instructions above"
    }}
    """

    response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt
)
    raw = response.text.strip()

    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw.strip())