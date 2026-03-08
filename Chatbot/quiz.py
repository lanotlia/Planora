import google.generativeai as genai
import os
import json

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel("gemini-1.5-flash")


def extract_text_from_material(material: str) -> str:
    """Clean and truncate material to fit context window"""
    # Gemini flash handles ~100k tokens but we keep it reasonable
    return material[:8000] if len(material) > 8000 else material


def generate_questions(
    material: str,
    user_profile: dict,      # pass the whole profile, not individual flags
    subject_name: str,
    n_questions: int = 5
) -> list:
    user_category = user_profile.get("user_category", "uni_student")
    struggle      = user_profile.get("struggle", "understanding")
    current_level = user_profile.get("current_level")

    # Question count adjustments per condition
    if user_profile.get("has_adhd") or user_profile.get("has_anxiety"):
        n_questions = min(n_questions, 4)   # fewer questions for ADHD and anxiety
    if user_profile.get("has_dyslexia"):
        n_questions = min(n_questions, 4)   # fewer but more accessible questions
    """
    Generate questions tailored to user category and learning profile.
    Returns a list of question objects.
    """

    material = extract_text_from_material(material)

    # ── Build category-specific question instructions ─────────────────────────
    if user_category == "language_learner":
        question_style = f"""
        The student is a {current_level or "intermediate"} language learner.
        Generate a MIX of these question types:
        - Vocabulary: "What does [word from the material] mean?"
        - Translation: "Translate this sentence into English: [sentence]"
        - Fill in the blank: "[Sentence with gap] — fill in the missing word"
        - Usage: "Use the word [word] in a sentence of your own"
        - Comprehension: A question about the meaning of a passage

        Weight vocabulary and fill-in-the-blank questions most heavily.
        Use words and sentences directly from the material provided.
        """

    elif user_category == "cert_candidate":
        question_style = """
        Generate exam-style questions similar to professional certification exams:
        - Multiple choice questions with 4 options (label them A, B, C, D)
        - Scenario-based questions: "In this situation, what would you do..."
        - Definition questions: "Define [term] in your own words"
        - Application questions: "How would you apply [concept] to [scenario]"

        Make questions specific, precise and exam-realistic.
        Always include the correct answer in the question object.
        """

    elif user_category == "uni_student":
        question_style = """
        Generate university-level questions that test genuine understanding:
        - Concept questions: "Explain [concept] in your own words"
        - Application: "How does [concept] apply to [scenario from material]"
        - Compare/contrast: "What is the difference between [x] and [y]"
        - Critical thinking: "Why does [phenomenon] happen?"

        Avoid surface-level recall. Prioritise understanding over memorisation.
        """

    elif user_category == "self_study":
        question_style = """
        Generate practical comprehension questions:
        - "Explain [concept] as if teaching someone else"
        - "What is the main purpose of [concept/tool/idea]?"
        - "Give an example of how you would use [concept]"
        - "What would happen if [condition changed]?"

        Focus on practical understanding and application.
        """

    # ── Modify for ADHD ───────────────────────────────────────────────────────
    adhd_note = ""
    if user_profile.get("has_adhd"):
        adhd_note = """
        IMPORTANT: Keep each question SHORT and direct.
        No long preambles. One concept per question maximum.
        """
    # ── Dyslexia accommodations ───────────────────────────────────────────────
    dyslexia_note = ""
    if user_profile.get("has_dyslexia"):
        dyslexia_note = """
        IMPORTANT — this student has dyslexia:
        - Keep question text short and direct. One sentence per question maximum.
        - Prefer multiple choice questions over open-ended written responses.
        - Avoid questions that require long written answers.
        - Use simple, common words. No unnecessarily complex vocabulary in the question itself.
        - Accept answers that are conceptually correct even if spelling is imperfect.
        - Never ask the student to spell or write out long passages.
        """

    # ── Autism accommodations ─────────────────────────────────────────────────
    autism_note = ""
    if user_profile.get("has_autism"):
        autism_note = """
        IMPORTANT — this student prefers clear structure:
        - Be completely explicit and literal. No figurative language or idioms.
        - Each question must have one clear unambiguous answer.
        - Avoid hypothetical or open-ended scenario questions.
        - Do not use phrases like "what do you think" or "in your opinion."
        - Prefer definition, factual recall, and specific application questions.
        - State exactly what kind of answer you want: "Give one example" not "can you think of an example?"
        """

    # ── Anxiety accommodations ────────────────────────────────────────────────
    anxiety_note = ""
    if user_profile.get("has_anxiety"):
        anxiety_note = """
        IMPORTANT — this student experiences study anxiety:
        - Frame questions as curiosity not testing. "What do you remember about..." 
          not "What is the correct answer for..."
        - Avoid words like "test", "correct", "wrong", "fail" in question text.
        - Make questions feel like a conversation, not an exam.
        - Multiple choice is less anxiety-inducing than open-ended for this student.
        """

    # ── Struggle-aware modifications ──────────────────────────────────────────
    struggle_note = ""
    if struggle == "understanding":
        struggle_note = "Weight questions toward conceptual explanation over facts."
    elif struggle == "remembering":
        struggle_note = "Include more definition and recall questions to strengthen memory."
    elif struggle in ["vocab_doesnt_stick", "grammar_confusion"]:
        struggle_note = "Focus heavily on vocabulary retention and correct usage."

    # ── Final prompt ──────────────────────────────────────────────────────────
    prompt = f"""
    You are a study coach testing a student on material they just finished studying.
    Subject: {subject_name}
    
    {question_style}
    {adhd_note}
    {dyslexia_note}
    {autism_note}
    {anxiety_note}
    {struggle_note}
    
    Based on the following study material, generate exactly {n_questions} questions.
    ... rest of prompt unchanged
    
    
    Return ONLY a JSON array. No extra text, no markdown, no explanation.
    Format:
    [
      {{
        "id": 1,
        "question": "The question text here",
        "type": "concept/vocabulary/multiple_choice/fill_blank/application",
        "options": ["A. option", "B. option", "C. option", "D. option"],  // only for multiple choice, else null
        "correct_answer": "The correct answer",
        "follow_up": "A short follow-up hint if they get it wrong"
      }}
    ]
    
    Study material:
    {material}
    """

    response = gemini.generate_content(prompt)
    raw = response.text.strip()

    # Strip markdown if Gemini adds it
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    questions = json.loads(raw.strip())
    return questions


def evaluate_answer(
    question: str,
    correct_answer: str,
    user_answer: str,
    question_type: str,
    user_category: str
) -> dict:
    """
    Evaluate a user's answer and return score + feedback.
    Uses Gemini to handle natural language answers fairly.
    """

    prompt = f"""
    You are a fair and encouraging study coach evaluating a student's answer.
    
    Question: {question}
    Question type: {question_type}
    Correct answer: {correct_answer}
    Student's answer: {user_answer}
    
    Evaluate the answer and respond ONLY with a JSON object:
    {{
      "score": 0-3,
      "correct": true/false,
      "feedback": "Brief, encouraging feedback. If wrong, explain why without being harsh.",
      "partial": true/false
    }}
    
    Scoring guide:
    - 3: Fully correct, shows clear understanding
    - 2: Mostly correct, minor gaps
    - 1: Partially correct, right direction but missing key points
    - 0: Incorrect or shows fundamental misunderstanding
    
    Be generous with partial credit for natural language answers.
    For multiple choice, only award 3 or 0.
    Keep feedback under 2 sentences. Always end on an encouraging note.
    """

    response = gemini.generate_content(prompt)
    raw = response.text.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw.strip())