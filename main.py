import sys
import os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from groq import Groq
from dotenv import load_dotenv
import gdown
import json

from model.predict import predict
from chatbot.checkin import CheckInSession
from database.supabase import (
    create_user, get_user, update_user,
    create_subject, get_subjects, get_subject,
    save_recommendations, get_recommendations,
    save_session, get_sessions,
    save_flashcards, get_due_flashcards, update_flashcard_after_review,
    save_quiz_results
)

load_dotenv()

# ── Groq client — single instance for all endpoints ──────────────────────────
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def _call_llm(prompt: str) -> str:
    """Single helper for all LLM calls in main.py"""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────────────────

def download_model_artifacts():
    os.makedirs("model/artifacts", exist_ok=True)
    files = {
        "model/artifacts/models.pkl":        "17OQb35a69y8RZ_mUfZLT9oVGzRv5IpAW",
        "model/artifacts/encoders.pkl":      "1rKLQj2WAJT5KCQvUy74mCSlno7u69CCi",
        "model/artifacts/feature_cols.json": "1qxM29f_35YWCyz7KSpg5VAr1MsrysXpL"
    }
    for path, file_id in files.items():
        if not os.path.exists(path):
            print(f"Downloading {path}...")
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}",
                path,
                quiet=False
            )
            print(f"{path} ready.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    download_model_artifacts()
    yield

app = FastAPI(
    title="Planora API",
    description="Personalised study coaching for every kind of learner",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_sessions: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────────────────────────

class UserProfile(BaseModel):
    user_name:       Optional[str] = "there"
    user_category:   str
    has_adhd:        int = 0
    has_dyslexia:    int = 0
    has_autism:      int = 0
    has_anxiety:     int = 0
    attention_span:  str
    sleep_hours:     float
    daily_study_hrs: float
    learning_style:  str
    peak_focus_time: str
    study_env:       str
    struggle:        str
    current_level:   Optional[str] = "not_applicable"
    prior_attempt:   Optional[str] = "not_applicable"
    success_goal:    Optional[str] = ""


class SubjectProfile(BaseModel):
    subject_name:  str
    content_type:  str
    memory_load:   str
    difficulty:    int
    has_deadline:  int
    days_to_exam:  Optional[int] = 999


class CheckInStart(BaseModel):
    user_id:      str
    user_profile: dict
    subject_name: str
    material:     Optional[str] = None


class CheckInMessage(BaseModel):
    user_id:  str
    message:  str


class FlashcardRequest(BaseModel):
    user_id:       str
    subject_id:    str
    subject_name:  str
    user_category: str
    material:      str


class FlashcardRating(BaseModel):
    user_id: str
    card_id: str
    rating:  int


class OnboardingProfile(BaseModel):
    user_name:       Optional[str] = "there"
    user_category:   str
    age:             Optional[int] = None
    has_adhd:        int = 0
    has_dyslexia:    int = 0
    has_autism:      int = 0
    has_anxiety:     int = 0
    attention_span:  str
    sleep_hours:     float
    daily_study_hrs: float
    learning_style:  str
    peak_focus_time: str
    study_env:       str
    struggle:        str
    current_level:   Optional[str] = "not_applicable"
    prior_attempt:   Optional[str] = "not_applicable"
    success_goal:    Optional[str] = ""


class SubjectCreate(BaseModel):
    user_id:      str
    subject_name: str
    content_type: str
    memory_load:  str
    difficulty:   int
    has_deadline: int
    days_to_exam: Optional[int] = 999
    exam_date:    Optional[str] = None


class RecommendRequest(BaseModel):
    user:       UserProfile
    subject:    SubjectProfile
    user_id:    Optional[str] = None
    subject_id: Optional[str] = None


class QARequest(BaseModel):
    user_id:       str
    question:      str
    subject_name:  str
    user_category: str
    context:       Optional[str] = None
    user_profile:  Optional[dict] = None


# ─────────────────────────────────────────────────────────────────────────────
# DOCS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Planora API Docs",
        swagger_js_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css",
    )

@app.get("/api-info", response_class=HTMLResponse)
async def api_info():
    return """
    <html><body>
        <h1>Planora API</h1>
        <ul>
            <li>POST /users</li>
            <li>POST /subjects</li>
            <li>GET  /subjects/{user_id}</li>
            <li>POST /recommend</li>
            <li>POST /checkin/start</li>
            <li>POST /checkin/message</li>
            <li>DELETE /checkin/{user_id}</li>
            <li>POST /flashcards/generate</li>
            <li>GET  /flashcards/due</li>
            <li>POST /flashcards/rate</li>
            <li>POST /qa</li>
        </ul>
    </body></html>
    """


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "Planora API is running",
        "version": "1.0.0",
        "docs": "https://planora-1-87zb.onrender.com/docs"
    }

@app.get("/ping")
async def ping():
    return {"status": "alive", "api": "Planora"}


# ─────────────────────────────────────────────────────────────────────────────
# USER ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/users")
async def create_new_user(profile: OnboardingProfile):
    try:
        user = create_user(profile.dict())
        return {"user_id": user["id"], "user": user}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/{user_id}")
async def get_user_profile(user_id: str):
    try:
        user = get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# SUBJECT ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/subjects")
async def add_subject(data: SubjectCreate):
    try:
        subject = create_subject(data.dict())
        return {"subject_id": subject["id"], "subject": subject}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/subjects/{user_id}")
async def list_subjects(user_id: str):
    try:
        subjects = get_subjects(user_id)
        return {"subjects": subjects, "count": len(subjects)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/recommend")
async def get_recommendations_endpoint(data: RecommendRequest):
    try:
        result = predict(data.user.dict(), data.subject.dict())

        if data.user_id and data.subject_id:
            save_recommendations(
                user_id         = data.user_id,
                subject_id      = data.subject_id,
                recommendations = result["recommendations"],
                session_meta    = {
                    "session_length": result["session_length"],
                    "daily_sessions": result["daily_sessions"]
                }
            )

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# CHECK-IN ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/checkin/start")
async def start_checkin(data: CheckInStart):
    try:
        session = CheckInSession(
            user_profile = data.user_profile,
            subject_name = data.subject_name,
            material     = data.material
        )
        opening_message = session.start()
        active_sessions[data.user_id] = session

        return {
            "message": opening_message,
            "stage":   "checkin",
            "user_id": data.user_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/checkin/message")
async def send_checkin_message(data: CheckInMessage):
    session = active_sessions.get(data.user_id)

    if not session:
        raise HTTPException(
            status_code=404,
            detail="No active session found. Call /checkin/start first."
        )

    try:
        if session.stage == "checkin":
            response = session.process_checkin_response(data.message)
            return {"message": response, "stage": session.stage}

        elif session.stage == "material_prompt":
            response = session.process_material_response(data.message)
            return {"message": response, "stage": session.stage}

        elif session.stage == "quiz":
            result = session.process_quiz_answer(data.message)

            if isinstance(result, tuple):
                message, summary = result
                active_sessions.pop(data.user_id, None)

                saved_session = None
                try:
                    subject_id = session.user_profile.get("subject_id")
                    if subject_id:
                        saved_session = save_session(
                            user_id         = data.user_id,
                            subject_id      = subject_id,
                            summary         = summary,
                            checkin_signals = session.checkin_signals
                        )
                        if saved_session:
                            save_quiz_results(
                                session_id   = saved_session["id"],
                                user_id      = data.user_id,
                                quiz_results = session.quiz_results
                            )
                            if summary.get("flashcards"):
                                save_flashcards(
                                    user_id    = data.user_id,
                                    subject_id = subject_id,
                                    session_id = saved_session["id"],
                                    flashcards = summary["flashcards"]
                                )
                except Exception as db_error:
                    print(f"DB save error: {db_error}")

                return {
                    "message":    message,
                    "stage":      "complete",
                    "summary":    summary,
                    "session_id": saved_session["id"] if saved_session else None,
                    "session":    session.get_session_data()
                }

            return {"message": result, "stage": "quiz"}

        else:
            active_sessions.pop(data.user_id, None)
            return {"message": "Session complete.", "stage": "complete"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/checkin/{user_id}")
async def end_checkin(user_id: str):
    if user_id in active_sessions:
        active_sessions.pop(user_id)
        return {"message": "Session ended successfully."}
    return {"message": "No active session found for this user."}


# ─────────────────────────────────────────────────────────────────────────────
# FLASHCARD ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/flashcards/generate")
async def create_flashcards(data: FlashcardRequest):
    try:
        material = data.material[:8000] if len(data.material) > 8000 else data.material

        prompt = f"""
        You are a study coach helping a {data.user_category} study {data.subject_name}.

        Based on the following course material, generate flashcards.

        Rules:
        - Each flashcard must have a clear, specific question on the front
        - The answer must be concise — no longer than 3 sentences
        - Focus on key concepts, definitions, formulas, and important relationships
        - Do not generate vague or overly broad questions
        - Generate between 5 and 15 flashcards depending on how much material is provided

        Return ONLY a valid JSON array. No extra text. No markdown.

        [
          {{
            "front": "question here",
            "back": "answer here",
            "difficulty": "easy/medium/hard"
          }}
        ]

        Course material:
        {material}
        """

        raw = _call_llm(prompt)

        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]

        flashcards = json.loads(raw.strip())

        enriched = []
        for card in flashcards:
            enriched.append({
                "user_id":        data.user_id,
                "subject_id":     data.subject_id,
                "front":          card["front"],
                "back":           card["back"],
                "difficulty":     card["difficulty"],
                "interval_days":  1,
                "ease_factor":    2.5,
                "next_review":    "today",
                "times_reviewed": 0,
                "last_rating":    None
            })

        return {"flashcards": enriched, "count": len(enriched)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/flashcards/due")
async def get_due_flashcards_endpoint(user_id: str, subject_id: str, today: str = None):
    try:
        review_date = today or datetime.now().strftime("%Y-%m-%d")
        cards = get_due_flashcards(user_id, subject_id, review_date)
        return {"flashcards": cards, "count": len(cards), "date": review_date}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/flashcards/rate")
async def rate_flashcard(data: FlashcardRating):
    try:
        from supabase import create_client
        sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
        card = sb.table("flashcards").select("*").eq("id", data.card_id).execute()

        if not card.data:
            raise HTTPException(status_code=404, detail="Flashcard not found")

        current = card.data[0]
        new_interval, new_ease, next_date = calculate_sm2(
            rating        = data.rating,
            interval_days = current["interval_days"],
            ease_factor   = current["ease_factor"]
        )

        update_flashcard_after_review(
            card_id          = data.card_id,
            rating           = data.rating,
            new_interval     = new_interval,
            new_ease         = new_ease,
            next_review_date = next_date
        )

        return {
            "card_id":           data.card_id,
            "new_interval_days": new_interval,
            "new_ease_factor":   new_ease,
            "next_review_date":  next_date,
            "message":           f"Next review scheduled for {next_date}"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def calculate_sm2(rating: int, interval_days: int, ease_factor: float) -> tuple:
    quality  = rating - 1
    new_ease = ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    new_ease = max(1.3, new_ease)

    if quality < 3:
        new_interval = 1
    elif interval_days == 1:
        new_interval = 3
    elif interval_days == 3:
        new_interval = 7
    else:
        new_interval = round(interval_days * new_ease)

    next_date = (datetime.now() + timedelta(days=new_interval)).strftime("%Y-%m-%d")
    return new_interval, round(new_ease, 2), next_date


# ─────────────────────────────────────────────────────────────────────────────
# STUDY Q&A ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/qa")
async def study_qa(data: QARequest):
    try:
        context_block = (
            f"\nUse this material as your primary reference:\n{data.context[:4000]}"
            if data.context else ""
        )

        profile      = data.user_profile or {}
        has_adhd     = profile.get("has_adhd", 0)
        has_dyslexia = profile.get("has_dyslexia", 0)
        has_autism   = profile.get("has_autism", 0)
        has_anxiety  = profile.get("has_anxiety", 0)

        condition_note = ""
        if has_adhd:
            condition_note = """
            Response style: Short paragraphs, max 3 sentences each.
            Get to the answer immediately. Maximum 3 paragraphs total.
            """
        elif has_dyslexia:
            condition_note = """
            Response style: Simple everyday words only.
            Very short sentences. Maximum 2 short paragraphs.
            """
        elif has_autism:
            condition_note = """
            Response style: Completely literal and explicit. No idioms.
            Definition first, then explanation. Maximum 3 paragraphs.
            """
        elif has_anxiety:
            condition_note = """
            Response style: Calm, reassuring tone.
            Frame as exploration not examination. Maximum 3 short paragraphs.
            """

        prompt = f"""
        You are a knowledgeable study coach helping a {data.user_category}
        with {data.subject_name}.

        The student asks: "{data.question}"
        {context_block}

        {condition_note}

        Answer clearly and helpfully:
        - Explain at the right level for a {data.user_category}
        - If concept question: explain the idea first then give an example
        - If calculation question: show the steps clearly
        - If language question: give the answer with an example in context
        - Stay focused — no unnecessary padding
        - If unsure about something, say so honestly
        - Maximum 4 short paragraphs
        """

        answer = _call_llm(prompt)

        return {
            "question": data.question,
            "answer":   answer,
            "subject":  data.subject_name,
            "user_id":  data.user_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))