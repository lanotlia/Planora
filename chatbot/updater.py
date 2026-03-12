import google.generativeai as genai
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel("gemini-2.0-flash")

def get_client():
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def extract_session_signals(checkin_text: str, user_category: str) -> dict:
    """
    Extract structured signals from the user's natural language check-in
    response. These signals are saved to Supabase and used to update the
    user's profile and adjust future recommendations over time.
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

    response = get_client().models.generate_content(
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


def generate_session_summary(
    checkin_signals: dict,
    quiz_results: list,
    user_profile: dict,
    subject_name: str
) -> dict:
    """
    Generate a complete session summary object.
    Calculates quiz performance, schedules next review using spaced
    repetition logic, and generates a personalised encouragement message
    adapted to the user's learning profile and any conditions they have.
    """

    total_score   = sum(r.get("score", 0) for r in quiz_results)
    max_score     = len(quiz_results) * 3
    score_pct     = round((total_score / max_score * 100) if max_score > 0 else 0)
    correct_count = sum(1 for r in quiz_results if r.get("correct"))

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
                                score_pct=score_pct,
                                sentiment=checkin_signals.get("sentiment", "neutral"),
                                success_goal=user_profile.get("success_goal", ""),
                                user_profile=user_profile
                            )
    }

    return summary


def _calculate_next_review(score_pct: int) -> str:
    """
    Use spaced repetition logic to schedule the next review date.
    Poor performance means sooner review.
    Strong performance means longer interval before next review.

    Based on the SM-2 spaced repetition algorithm intervals.
    """

    today = datetime.now()

    if score_pct >= 80:
        days = 7      # strong understanding — review in one week
    elif score_pct >= 60:
        days = 3      # moderate — review in 3 days
    elif score_pct >= 40:
        days = 2      # struggling — review in 2 days
    else:
        days = 1      # needs immediate review tomorrow

    next_date = today + timedelta(days=days)
    return next_date.strftime("%Y-%m-%d")


def _generate_encouragement(
    score_pct: int,
    sentiment: str,
    success_goal: str,
    user_profile: dict
) -> str:
    """
    Generate a personalised closing encouragement message.
    The tone, length, and framing adapt to the user's learning profile
    and any conditions they have. Each condition gets a distinct approach
    grounded in what research says those learners actually respond well to.
    """

    has_adhd     = user_profile.get("has_adhd", 0)
    has_dyslexia = user_profile.get("has_dyslexia", 0)
    has_autism   = user_profile.get("has_autism", 0)
    has_anxiety  = user_profile.get("has_anxiety", 0)

    # ── Autism: direct, factual, no motivational fluff ────────────────────────
    # Autistic learners often find performative warmth confusing or irritating.
    # They respond better to honest, direct, specific feedback.
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
        - No motivational language. No "you've got this" or similar phrases.
        - No exclamation marks.
        - No idioms or metaphors.
        - Be honest. If the score was low, say what needs work directly.
        - Do not say "great job" or any equivalent phrase.

        Good example style:
        "You answered 3 out of 5 questions correctly on Newton's Laws.
        Reviewing the two missed concepts before your next session will
        strengthen your understanding of the topic."
        """

    # ── Dyslexia: short, simple, warm, effort-focused ─────────────────────────
    # Dyslexic learners often carry years of academic self-doubt.
    # Focus on effort and progress, not performance metrics.
    # Keep it short because long messages are harder to process.
    elif has_dyslexia:
        prompt = f"""
        Write a closing message for a student with dyslexia.

        Their quiz result: {score_pct}%
        Their goal: "{success_goal or 'improve their studies'}"

        Rules:
        - Maximum 2 short sentences. This is a hard limit.
        - Use only simple, everyday words. Nothing over 3 syllables if possible.
        - Focus on their effort and progress, not the score.
        - Warm and encouraging tone.
        - Do not mention the score percentage directly.
        - Connect to their goal if it fits naturally in one sentence.
        - No complex sentence structures.
        """

    # ── ADHD: short, punchy, forward-looking ─────────────────────────────────
    # ADHD learners respond to momentum and clear next steps.
    # Long reflective messages lose their attention. Be brief and specific.
    elif has_adhd:
        prompt = f"""
        Write a closing message for a student with ADHD.

        Their quiz result: {score_pct}%
        Their goal: "{success_goal or 'improve their studies'}"

        Rules:
        - Maximum 2 sentences. Short sentences only — under 15 words each.
        - Be forward-looking. What is the very next step?
        - Energetic but not over the top.
        - Reference their goal if possible.
        - No long reflections or looking backward.
        - Make it feel like a quick handoff to what comes next.
        """

    # ── Anxiety: effort-focused, no score emphasis ───────────────────────────
    # Anxiety around studying is often tied to performance evaluation.
    # Remove the score entirely from the framing. Focus on process.
    elif has_anxiety:
        prompt = f"""
        Write a closing message for a student who experiences study anxiety.

        Their mood after the session: {sentiment}
        Their goal: "{success_goal or 'improve their studies'}"

        Rules:
        - Do not mention the score or percentage at all.
        - Focus entirely on their effort and the act of showing up.
        - Normalise difficulty if the sentiment was negative.
        - Warm and reassuring. Not patronising.
        - Maximum 2 sentences.
        - Connect to their goal if it fits naturally.
        - Avoid words like "test", "score", "performance", "result."
        """

    # ── Standard: everyone else ──────────────────────────────────────────────
    else:
        prompt = f"""
        Write a closing message for a student who just finished a study
        session and a quiz.

        Their quiz score: {score_pct}%
        Their mood: {sentiment}
        Their goal: "{success_goal or 'improve their studies'}"

        Rules:
        - Be genuine and specific. Avoid generic phrases.
        - Do not use "Great job!", "Well done!", or "Amazing!" — be more specific.
        - If score is low, focus on the path forward not the number.
        - If score is high, connect it meaningfully to their stated goal.
        - Reference their goal if possible.
        - Maximum 2 sentences.
        """

    response = gemini.generate_content(prompt)
    return response.text.strip()