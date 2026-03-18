from groq import Groq
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def _call_llm(prompt: str) -> str:
    """Single helper for all LLM calls"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def extract_session_signals(checkin_text: str, user_category: str) -> dict:
    """
    Extract structured signals from the user's natural language check-in response.
    These signals are saved to Supabase and used to update the user's profile
    and adjust future recommendations over time.
    """

    prompt = f"""
    A student just finished a study session and wrote this check-in message:
    "{checkin_text}"

    Their user category: {user_category}

    Extract signals from this message and return ONLY a valid JSON object.
    No extra text. No markdown. Just the JSON.

    {{
      "focus_quality": 1-5,
      "energy_level": 1-5,
      "session_difficulty": 1-5,
      "motivation": 1-5,
      "completed_session": true/false,
      "mentioned_distraction": true/false,
      "mentioned_confusion": true/false,
      "mentioned_tiredness": true/false,
      "mentioned_anxiety": true/false,
      "sentiment": "positive/neutral/negative",
      "needs_plan_adjustment": true/false,
      "notes": "Any specific concern worth flagging in one sentence, or null"
    }}

    Instructions:
    - Infer scores from tone and content even if not explicitly stated.
    - If the message is very short or vague, default numeric scores to 3.
    - "needs_plan_adjustment" should be true if the student reports
      consistent difficulty, burnout, or major focus problems.
    - "mentioned_anxiety" should be true if they express stress, worry,
      feeling overwhelmed, or nervousness about the material or upcoming exams.
    """

    raw = _call_llm(prompt)

    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw.strip())


def generate_session_summary(
    checkin_signals: dict,
    quiz_results: list,
    user_profile: dict,
    subject_name: str
) -> dict:
    """
    Generate a complete session summary object.
    Calculates quiz performance, schedules next review using spaced
    repetition logic, and generates a personalised encouragement message.
    """

    total_score   = sum(r.get("score", 0) for r in quiz_results)
    max_score     = len(quiz_results) * 3
    score_pct     = round((total_score / max_score * 100) if max_score > 0 else 0)
    correct_count = sum(1 for r in quiz_results if r.get("correct"))

    if score_pct >= 80:
        understanding = "strong"
    elif score_pct >= 50:
        understanding = "moderate"
    else:
        understanding = "needs_review"

    summary = {
        "subject":          subject_name,
        "date":             datetime.now().isoformat(),
        "quiz_score":       score_pct,
        "correct_answers":  correct_count,
        "total_questions":  len(quiz_results),
        "understanding":    understanding,
        "focus_quality":    checkin_signals.get("focus_quality", 3),
        "energy_level":     checkin_signals.get("energy_level", 3),
        "needs_review":     understanding == "needs_review",
        "next_review_date": _calculate_next_review(score_pct),
        "encouragement":    _generate_encouragement(
                                score_pct    = score_pct,
                                sentiment    = checkin_signals.get("sentiment", "neutral"),
                                success_goal = user_profile.get("success_goal", ""),
                                user_profile = user_profile
                            )
    }

    return summary


def _calculate_next_review(score_pct: int) -> str:
    """
    Schedule next review date using spaced repetition logic.
    Poor performance = sooner review.
    """
    today = datetime.now()

    if score_pct >= 80:
        days = 7
    elif score_pct >= 60:
        days = 3
    elif score_pct >= 40:
        days = 2
    else:
        days = 1

    return (today + timedelta(days=days)).strftime("%Y-%m-%d")


def _generate_encouragement(
    score_pct: int,
    sentiment: str,
    success_goal: str,
    user_profile: dict
) -> str:
    """
    Generate a personalised closing encouragement message.
    Tone and framing adapt to the user's learning profile and conditions.
    """

    has_adhd     = user_profile.get("has_adhd", 0)
    has_dyslexia = user_profile.get("has_dyslexia", 0)
    has_autism   = user_profile.get("has_autism", 0)
    has_anxiety  = user_profile.get("has_anxiety", 0)

    if has_autism:
        prompt = f"""
        Write a closing message for a student who prefers direct, literal
        communication without social pleasantries.

        Their quiz result: {score_pct}%
        Their goal: "{success_goal or 'improve their studies'}"

        Rules:
        - Write exactly 2 sentences. No more, no less.
        - Sentence 1: One factual, specific observation about their performance.
        - Sentence 2: One concrete, specific next action they should take.
        - No motivational language. No exclamation marks. No idioms.
        - Be honest. Do not say "great job" or any equivalent phrase.
        """

    elif has_dyslexia:
        prompt = f"""
        Write a closing message for a student with dyslexia.

        Their goal: "{success_goal or 'improve their studies'}"

        Rules:
        - Maximum 2 short sentences. Hard limit.
        - Use only simple, everyday words. Nothing over 3 syllables.
        - Focus on effort and progress, not the score.
        - Do not mention the score percentage.
        - Warm and encouraging tone.
        """

    elif has_adhd:
        prompt = f"""
        Write a closing message for a student with ADHD.

        Their quiz result: {score_pct}%
        Their goal: "{success_goal or 'improve their studies'}"

        Rules:
        - Maximum 2 sentences. Under 15 words each.
        - Forward-looking. What is the very next step?
        - Energetic but not over the top.
        - Reference their goal if possible.
        """

    elif has_anxiety:
        prompt = f"""
        Write a closing message for a student who experiences study anxiety.

        Their mood: {sentiment}
        Their goal: "{success_goal or 'improve their studies'}"

        Rules:
        - Do not mention the score or percentage at all.
        - Focus on effort and the act of showing up.
        - Normalise difficulty if sentiment was negative.
        - Warm and reassuring. Not patronising.
        - Maximum 2 sentences.
        - Avoid words like test, score, performance, result.
        """

    else:
        prompt = f"""
        Write a closing message for a student who just finished a study session.

        Their quiz score: {score_pct}%
        Their mood: {sentiment}
        Their goal: "{success_goal or 'improve their studies'}"

        Rules:
        - Be genuine and specific. No generic phrases.
        - Do not use "Great job!", "Well done!", or "Amazing!"
        - If score is low, focus on the path forward not the number.
        - If score is high, connect it to their stated goal.
        - Maximum 2 sentences.
        """

    return _call_llm(prompt)