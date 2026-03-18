from groq import Groq
import os
import json
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def _call_llm(prompt: str) -> str:
    """Single helper for all LLM calls"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def extract_text_from_material(material: str) -> str:
    return material[:8000] if len(material) > 8000 else material


def generate_questions(
    material: str,
    user_profile: dict,
    subject_name: str,
    n_questions: int = 5
) -> list:

    material      = extract_text_from_material(material)
    user_category = user_profile.get("user_category", "uni_student")
    struggle      = user_profile.get("struggle", "understanding")
    current_level = user_profile.get("current_level")
    has_adhd      = user_profile.get("has_adhd", 0)
    has_dyslexia  = user_profile.get("has_dyslexia", 0)
    has_autism    = user_profile.get("has_autism", 0)
    has_anxiety   = user_profile.get("has_anxiety", 0)

    if has_adhd or has_anxiety or has_dyslexia:
        n_questions = min(n_questions, 4)

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

    if has_autism:
        question_style = f"""
        This student needs completely explicit and unambiguous questions
        about {subject_name}.
        ONLY generate these question types:
        - Definition: "What is [term]?" — one clear factual answer
        - True or False: "True or false: [statement]"
        - Multiple choice with 4 options — one is clearly and objectively correct
        - Specific recall: "What year did [event] happen?"
        NEVER generate open-ended or opinion questions.
        Every question must have exactly one objectively correct answer.
        """

    adhd_note = ""
    if has_adhd:
        adhd_note = """
        ADHD: Keep each question SHORT and direct. One sentence maximum.
        No long preambles. One concept per question only.
        """

    dyslexia_note = ""
    if has_dyslexia:
        dyslexia_note = """
        Dyslexia: Keep question text short. Prefer multiple choice.
        Use simple everyday words. Never ask the student to copy or transcribe text.
        """

    anxiety_note = ""
    if has_anxiety:
        anxiety_note = """
        Anxiety: Frame questions as curiosity not testing.
        Use "What do you remember about..." not "What is the correct answer for..."
        Avoid the words test, correct, wrong, fail, must in question text.
        """

    struggle_note = ""
    if struggle == "understanding":
        struggle_note = "Weight questions toward conceptual explanation and reasoning."
    elif struggle == "remembering":
        struggle_note = "Include more definition and active recall questions."
    elif struggle in ["vocab_doesnt_stick", "grammar_confusion"]:
        struggle_note = "Focus heavily on vocabulary retention and correct usage."
    elif struggle == "cant_produce":
        struggle_note = "Include questions that require the student to produce language."

    prompt = f"""
    You are a study coach reviewing material with a student who just
    finished studying. Generate questions that help them consolidate learning.

    Subject: {subject_name}

    {question_style}
    {adhd_note}
    {dyslexia_note}
    {anxiety_note}
    {struggle_note}

    Based on the following study material, generate exactly {n_questions} questions.

    CRITICAL: Return ONLY a valid JSON array. No extra text. No markdown.

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

    Notes:
    - "options" only for multiple_choice and true_false, otherwise null
    - "correct_answer" must always be filled in
    - "follow_up" should hint without giving away the answer

    Study material:
    {material}
    """

    raw = _call_llm(prompt)

    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw.strip())


def evaluate_answer(
    question: str,
    correct_answer: str,
    user_answer: str,
    question_type: str,
    user_profile: dict
) -> dict:

    has_dyslexia = user_profile.get("has_dyslexia", 0)
    has_autism   = user_profile.get("has_autism", 0)
    has_anxiety  = user_profile.get("has_anxiety", 0)
    has_adhd     = user_profile.get("has_adhd", 0)

    if has_autism:
        feedback_style = """
        Be completely direct. Say "That is correct." or "That is not correct."
        Then one factual sentence if wrong. No praise. Factual only. 2 sentences max.
        """
    elif has_dyslexia:
        feedback_style = """
        Use very simple short words. Maximum 2 short sentences.
        Do not mention spelling errors. Evaluate meaning only.
        """
    elif has_adhd:
        feedback_style = """
        Be brief and direct. Maximum 2 sentences.
        Acknowledge correct answers and move on.
        Give key correction in one short sentence if wrong.
        """
    elif has_anxiety:
        feedback_style = """
        Never use: wrong, incorrect, failed, mistake.
        If wrong: normalise it first then explain.
        Example: "This one trips a lot of people up — the key thing is [correction]."
        Emphasise what they understood. Maximum 2 sentences.
        """
    else:
        feedback_style = """
        Brief and encouraging. Maximum 2 sentences.
        Acknowledge correct answers specifically.
        Explain incorrect answers clearly without being harsh.
        """

    spelling_note = ""
    if has_dyslexia:
        spelling_note = """
        Do not penalise spelling errors. Evaluate meaning and concept only.
        A misspelled answer showing correct understanding scores the same as perfect spelling.
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
    - 2: Mostly correct, minor gaps
    - 1: Partially correct, right direction but missing key points
    - 0: Incorrect or fundamental misunderstanding

    Be generous with partial credit for natural language answers.
    For multiple choice and true/false, only award 3 or 0.

    Respond ONLY with a valid JSON object. No extra text. No markdown.

    {{
      "score": 0-3,
      "correct": true/false,
      "partial": true/false,
      "feedback": "Your feedback here"
    }}
    """

    raw = _call_llm(prompt)

    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw.strip())


def generate_flashcards_for_session(
    material: str,
    user_profile: dict,
    subject_name: str,
    quiz_results: list,
    base_interval: int = 1
) -> list:

    material = material[:6000] if len(material) > 6000 else material

    wrong_concepts = []
    for result in quiz_results:
        if not result.get("correct") and result.get("question_id"):
            wrong_concepts.append(str(result.get("question_id")))

    wrong_note = ""
    if wrong_concepts:
        wrong_note = f"""
        PRIORITY: The student struggled with questions {', '.join(wrong_concepts)}.
        Generate at least 2-3 flashcards targeting those concepts specifically.
        """

    has_adhd     = user_profile.get("has_adhd", 0)
    has_dyslexia = user_profile.get("has_dyslexia", 0)

    format_note = ""
    if has_adhd or has_dyslexia:
        format_note = """
        Keep flashcard questions SHORT. One line maximum.
        Keep answers SHORT. One or two sentences maximum.
        Prefer simple vocabulary.
        """

    prompt = f"""
    You are creating flashcards for a student who just finished studying {subject_name}.
    These cards will appear before their next study session to reinforce memory.

    {wrong_note}
    {format_note}

    Rules:
    - Generate between 5 and 10 flashcards
    - Focus on key concepts, definitions, and relationships
    - Each card must be self-contained
    - Questions should be specific and answerable from memory
    - Answers should be concise — maximum 2 sentences
    - Vary question types: definitions, applications, comparisons, cause-effect

    Return ONLY a valid JSON array. No markdown. No extra text.
    [
      {{
        "front": "question here",
        "back": "answer here",
        "difficulty": "easy/medium/hard",
        "concept_tag": "short label for the concept this card covers"
      }}
    ]

    Study material:
    {material}
    """

    raw = _call_llm(prompt)

    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]

    cards = json.loads(raw.strip())

    today = datetime.now()
    enriched = []
    for i, card in enumerate(cards):
        is_priority = i < len(wrong_concepts)
        interval    = 1 if is_priority else base_interval

        enriched.append({
            "front":            card["front"],
            "back":             card["back"],
            "difficulty":       card["difficulty"],
            "concept_tag":      card.get("concept_tag", ""),
            "interval_days":    interval,
            "ease_factor":      2.5,
            "next_review_date": (today + timedelta(days=interval)).strftime("%Y-%m-%d"),
            "times_reviewed":   0,
            "last_rating":      None,
            "is_priority":      is_priority
        })

    return enriched


def _get_base_interval(score_pct: int) -> int:
    if score_pct >= 80:
        return 3
    elif score_pct >= 60:
        return 2
    else:
        return 1