import google.generativeai as genai
import os
import json
from datetime import datetime, timedelta

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel("gemini-1.5-flash")


def extract_session_signals(checkin_text: str, user_category: str) -> dict:
    """
    Extract structured signals from the user's natural language check-in response.
    These signals update the user profile over time.
    """

    prompt = f"""
    A student just finished a study session and wrote this check-in message:
    "{checkin_text}"
    
    Extract signals from this message. Return ONLY a JSON object:
    {{
      "focus_quality": 1-5,
      "energy_level": 1-5,
      "session_difficulty": 1-5,
      "motivation": 1-5,
      "completed_session": true/false,
      "mentioned_distraction": true/false,
      "mentioned_confusion": true/false,
      "mentioned_tiredness": true/false,
      "sentiment": "positive/neutral/negative",
      "needs_plan_adjustment": true/false,
      "notes": "Any specific concern worth flagging in one sentence or null"
    }}
    
    Infer scores from the tone and content even if not explicitly stated.
    If the message is very short or vague, default numeric scores to 3.
    """

    response = gemini.generate_content(prompt)
    raw = response.text.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
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
    Generate a session summary with adjusted recommendations if needed.
    """

    total_score    = sum(r.get("score", 0) for r in quiz_results)
    max_score      = len(quiz_results) * 3
    score_pct      = round((total_score / max_score * 100) if max_score > 0 else 0)
    correct_count  = sum(1 for r in quiz_results if r.get("correct"))

    # Determine understanding level from quiz performance
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
                                score_pct,
                                checkin_signals.get("sentiment", "neutral"),
                                user_profile.get("success_goal", "")
                            )
    }

    return summary


def _calculate_next_review(score_pct: int) -> str:
    """
    Use spaced repetition logic to schedule next review.
    Poor performance = sooner review.
    """
    today = datetime.now()

    if score_pct >= 80:
        days = 7       # strong understanding, review in a week
    elif score_pct >= 60:
        days = 3       # moderate, review in 3 days
    elif score_pct >= 40:
        days = 2       # struggling, review in 2 days
    else:
        days = 1       # needs immediate review tomorrow

    next_date = today + timedelta(days=days)
    return next_date.strftime("%Y-%m-%d")


def _generate_encouragement(
    score_pct: int,
    sentiment: str,
    success_goal: str
) -> str:
    """Generate a short personalised encouragement message"""

    prompt = f"""
    Write a single short encouraging message (2 sentences max) for a student who just 
    finished a study session.
    
    Their quiz score: {score_pct}%
    Their mood after the session: {sentiment}
    Their stated goal: "{success_goal or 'improve their studies'}"
    
    Rules:
    - Be genuine, not generic
    - Reference their goal if possible
    - If score is low, focus on effort and improvement not the score
    - If score is high, acknowledge it and connect it to their goal
    - Never use phrases like "Great job!" or "Well done!" — be more specific
    - Maximum 2 sentences
    """

    response = gemini.generate_content(prompt)
    return response.text.strip()