from supabase import create_client, Client
from dotenv import load_dotenv
import os

load_dotenv()

url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)


# ── USERS ─────────────────────────────────────────────────────────────────────

def create_user(profile: dict) -> dict:
    """Save a new user profile from onboarding"""
    result = supabase.table("users").insert(profile).execute()
    return result.data[0] if result.data else None


def get_user(user_id: str) -> dict:
    """Get a user profile by ID"""
    result = supabase.table("users").select("*").eq("id", user_id).execute()
    return result.data[0] if result.data else None


def update_user(user_id: str, updates: dict) -> dict:
    """Update a user profile — called when check-in signals suggest profile changes"""
    result = supabase.table("users").update(updates).eq("id", user_id).execute()
    return result.data[0] if result.data else None


# ── SUBJECTS ──────────────────────────────────────────────────────────────────

def create_subject(subject: dict) -> dict:
    """Save a new subject for a user"""
    result = supabase.table("subjects").insert(subject).execute()
    return result.data[0] if result.data else None


def get_subjects(user_id: str) -> list:
    """Get all active subjects for a user"""
    result = (
        supabase.table("subjects")
        .select("*")
        .eq("user_id", user_id)
        .eq("is_active", True)
        .execute()
    )
    return result.data or []


def get_subject(subject_id: str) -> dict:
    """Get a single subject by ID"""
    result = supabase.table("subjects").select("*").eq("id", subject_id).execute()
    return result.data[0] if result.data else None


# ── RECOMMENDATIONS ───────────────────────────────────────────────────────────

def save_recommendations(user_id: str, subject_id: str, recommendations: list, session_meta: dict) -> list:
    """Save technique recommendations for a user and subject"""
    rows = []
    for rec in recommendations:
        rows.append({
            "user_id":        user_id,
            "subject_id":     subject_id,
            "technique_id":   rec["id"],
            "technique_name": rec["name"],
            "rank":           rec["rank"],
            "confidence":     rec["confidence"],
            "how_to":         rec["how_to"],
            "why":            rec["why"],
            "session_length": session_meta.get("session_length", ""),
            "daily_sessions": session_meta.get("daily_sessions", ""),
        })

    result = supabase.table("recommendations").insert(rows).execute()
    return result.data or []


def get_recommendations(user_id: str, subject_id: str) -> list:
    """Get the latest recommendations for a user and subject"""
    result = (
        supabase.table("recommendations")
        .select("*")
        .eq("user_id", user_id)
        .eq("subject_id", subject_id)
        .order("rank")
        .execute()
    )
    return result.data or []


# ── STUDY SESSIONS ────────────────────────────────────────────────────────────

def save_session(user_id: str, subject_id: str, summary: dict, checkin_signals: dict) -> dict:
    """Save a completed check-in session"""
    row = {
        "user_id":              user_id,
        "subject_id":           subject_id,
        "checkin_text":         summary.get("checkin_text", ""),
        "focus_quality":        checkin_signals.get("focus_quality", 3),
        "energy_level":         checkin_signals.get("energy_level", 3),
        "session_difficulty":   checkin_signals.get("session_difficulty", 3),
        "motivation":           checkin_signals.get("motivation", 3),
        "sentiment":            checkin_signals.get("sentiment", "neutral"),
        "mentioned_distraction":checkin_signals.get("mentioned_distraction", False),
        "mentioned_confusion":  checkin_signals.get("mentioned_confusion", False),
        "mentioned_tiredness":  checkin_signals.get("mentioned_tiredness", False),
        "mentioned_anxiety":    checkin_signals.get("mentioned_anxiety", False),
        "needs_plan_adjustment":checkin_signals.get("needs_plan_adjustment", False),
        "quiz_score":           summary.get("quiz_score", 0),
        "correct_answers":      summary.get("correct_answers", 0),
        "total_questions":      summary.get("total_questions", 0),
        "understanding":        summary.get("understanding", ""),
        "next_review_date":     summary.get("next_review_date", ""),
        "notes":                checkin_signals.get("notes", ""),
    }

    result = supabase.table("study_sessions").insert(row).execute()
    return result.data[0] if result.data else None


def get_sessions(user_id: str, subject_id: str = None) -> list:
    """Get all sessions for a user, optionally filtered by subject"""
    query = supabase.table("study_sessions").select("*").eq("user_id", user_id)
    if subject_id:
        query = query.eq("subject_id", subject_id)
    result = query.order("created_at", desc=True).execute()
    return result.data or []


# ── FLASHCARDS ────────────────────────────────────────────────────────────────

def save_flashcards(user_id: str, subject_id: str, session_id: str, flashcards: list) -> list:
    """Save flashcards generated after a study session"""
    rows = []
    for card in flashcards:
        rows.append({
            "user_id":          user_id,
            "subject_id":       subject_id,
            "session_id":       session_id,
            "front":            card["front"],
            "back":             card["back"],
            "difficulty":       card["difficulty"],
            "concept_tag":      card.get("concept_tag", ""),
            "is_priority":      card.get("is_priority", False),
            "interval_days":    card.get("interval_days", 1),
            "ease_factor":      card.get("ease_factor", 2.5),
            "next_review_date": card.get("next_review_date", ""),
            "times_reviewed":   0,
            "last_rating":      None,
        })

    result = supabase.table("flashcards").insert(rows).execute()
    return result.data or []


def get_due_flashcards(user_id: str, subject_id: str, today: str) -> list:
    """Get all flashcards due for review today or earlier"""
    result = (
        supabase.table("flashcards")
        .select("*")
        .eq("user_id", user_id)
        .eq("subject_id", subject_id)
        .lte("next_review_date", today)   # less than or equal to today
        .order("is_priority", desc=True)  # priority cards first
        .order("next_review_date")
        .execute()
    )
    return result.data or []


def update_flashcard_after_review(card_id: str, rating: int, new_interval: int, new_ease: float, next_review_date: str) -> dict:
    """Update a flashcard's spaced repetition values after the user reviews it"""
    from datetime import datetime
    result = (
        supabase.table("flashcards")
        .update({
            "last_rating":      rating,
            "interval_days":    new_interval,
            "ease_factor":      new_ease,
            "next_review_date": next_review_date,
            "last_reviewed":    datetime.now().isoformat(),
            "times_reviewed":   supabase.table("flashcards").select("times_reviewed").eq("id", card_id).execute().data[0]["times_reviewed"] + 1
        })
        .eq("id", card_id)
        .execute()
    )
    return result.data[0] if result.data else None


# ── QUIZ RESULTS ──────────────────────────────────────────────────────────────

def save_quiz_results(session_id: str, user_id: str, quiz_results: list) -> list:
    """Save individual question results from a check-in session"""
    rows = []
    for result in quiz_results:
        rows.append({
            "session_id":     session_id,
            "user_id":        user_id,
            "question_id":    result.get("question_id"),
            "score":          result.get("score", 0),
            "correct":        result.get("correct", False),
            "partial":        result.get("partial", False),
            "feedback":       result.get("feedback", ""),
        })

    result = supabase.table("quiz_results").insert(rows).execute()
    return result.data or []