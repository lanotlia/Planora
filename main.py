from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
from model.predict import predict
from chatbot.checkin import CheckInSession
from dotenv import load_dotenv
import gdown
import os
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from google import genai
from datetime import datetime, timedelta

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# STARTUP — train model if artifacts don't exist
# ─────────────────────────────────────────────────────────────────────────────


def download_model_artifacts():
    os.makedirs("model/artifacts", exist_ok=True)
    
    files = {
        "model/artifacts/models.pkl":       "17OQb35a69y8RZ_mUfZLT9oVGzRv5IpAW",
        "model/artifacts/encoders.pkl":     "1rKLQj2WAJT5KCQvUy74mCSlno7u69CCi",
        "model/artifacts/feature_cols.json":"1qxM29f_35YWCyz7KSpg5VAr1MsrysXpL"
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



# Allows the mobile app to call the API without being blocked
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tighten this to your app's domain before going live
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store — one CheckInSession object per active user
# In production this would be Redis, for demo this is fine
active_sessions: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────────────────────────

class UserProfile(BaseModel):
    # Identity
    user_name:       Optional[str] = "there"
    user_category:   str                        # uni_student / language_learner / cert_candidate / self_study

    # Learning conditions — integers 0 or 1
    has_adhd:        int = 0
    has_dyslexia:    int = 0
    has_autism:      int = 0
    has_anxiety:     int = 0

    # Learning profile
    attention_span:  str                        # under_10 / 10_20 / 20_45 / 45_plus
    sleep_hours:     float
    daily_study_hrs: float
    learning_style:  str                        # read_write / visual / auditory / kinesthetic
    peak_focus_time: str                        # morning / afternoon / evening / late_night
    study_env:       str                        # quiet_home / noisy_home / library / cafe / varies
    struggle:        str                        # staying_focused / remembering / understanding etc.

    # Optional — populated depending on user category
    current_level:   Optional[str] = "not_applicable"   # language learners only
    prior_attempt:   Optional[str] = "not_applicable"   # cert candidates only
    success_goal:    Optional[str] = ""                 # open text goal from onboarding


class SubjectProfile(BaseModel):
    subject_name:  str
    content_type:  str           # theory / calculation / mixed / practical
    memory_load:   str           # high / medium / low
    difficulty:    int           # 1-5
    has_deadline:  int           # 0 or 1
    days_to_exam:  Optional[int] = 999


class CheckInStart(BaseModel):
    user_id:      str
    user_profile: dict           # full user profile dict
    subject_name: str
    material:     Optional[str] = None   # pasted text or extracted PDF text


class CheckInMessage(BaseModel):
    user_id:  str
    message:  str


class FlashcardRequest(BaseModel):
    user_id:       str
    subject_id:    str
    subject_name:  str
    user_category: str
    material:      str           # text to generate flashcards from


# Serve docs manually
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
    <html>
        <body>
            <h1>Planora API</h1>
            <h2>Endpoints</h2>
            <ul>
                <li>POST /recommend</li>
                <li>POST /checkin/start</li>
                <li>POST /checkin/message</li>
                <li>POST /flashcards/generate</li>
            </ul>
        </body>
    </html>
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
# RECOMMENDATION ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/recommend")
async def get_recommendations(user: UserProfile, subject: SubjectProfile):
    """
    Takes a user profile and a subject profile.
    Returns a personalised list of study technique recommendations
    with explanations, session length, and daily session count.
    """
    try:
        result = predict(user.dict(), subject.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# CHECK-IN ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/checkin/start")
async def start_checkin(data: CheckInStart):
    """
    Start a new check-in session for a user after a study session.
    Optionally accepts study material (text) to generate quiz questions from.
    Returns the opening message from the study coach chatbot.
    """
    try:
        session = CheckInSession(
            user_profile = data.user_profile,
            subject_name = data.subject_name,
            material     = data.material
        )
        opening_message = session.start()

        # Store session keyed by user_id
        # Overwrites any previous incomplete session for this user
        active_sessions[data.user_id] = session

        return {
            "message":  opening_message,
            "stage":    "checkin",
            "user_id":  data.user_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/checkin/message")
async def send_checkin_message(data: CheckInMessage):
    """
    Handle each message in the check-in conversation.
    Routes to the correct handler based on the current session stage:
      - checkin:  processes how-did-it-go, transitions to quiz
      - quiz:     evaluates answer, returns feedback + next question
                  or returns final summary when quiz is complete
      - complete: session is over
    """
    session = active_sessions.get(data.user_id)

    if not session:
        raise HTTPException(
            status_code=404,
            detail="No active session found. Call /checkin/start first."
        )

    try:
        # ── Stage: initial check-in ───────────────────────────────────────────
        if session.stage == "checkin":
            response = session.process_checkin_response(data.message)
            return {
                "message": response,
                "stage":   session.stage,
            }

        # ── Stage: quiz ───────────────────────────────────────────────────────
        elif session.stage == "quiz":
            result = session.process_quiz_answer(data.message)

            # process_quiz_answer returns a tuple when the quiz is complete
            if isinstance(result, tuple):
                message, summary = result

                # Clean up session from memory once complete
                active_sessions.pop(data.user_id, None)

                return {
                    "message":  message,
                    "stage":    "complete",
                    "summary":  summary,
                    "session":  session.get_session_data()
                }

            # Still in quiz — more questions remaining
            return {
                "message": result,
                "stage":   "quiz",
            }

        # ── Stage: complete ───────────────────────────────────────────────────
        else:
            active_sessions.pop(data.user_id, None)
            return {
                "message": "Session complete.",
                "stage":   "complete"
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/checkin/{user_id}")
async def end_checkin(user_id: str):
    """
    Manually end and clean up an active check-in session.
    Call this if the user closes the app mid-session.
    """
    if user_id in active_sessions:
        active_sessions.pop(user_id)
        return {"message": "Session ended successfully."}
    return {"message": "No active session found for this user."}


# ─────────────────────────────────────────────────────────────────────────────
# FLASHCARD ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/flashcards/generate")
async def create_flashcards(data: FlashcardRequest):
    """
    Generate flashcards from study material using Gemini.
    Returns flashcard objects enriched with spaced repetition starting values,
    ready to be saved directly to Supabase.
    """
    try:
        import json

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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

        response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=prompt
)
        raw = response.text.strip()

        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]

        flashcards = json.loads(raw.strip())

        # Attach spaced repetition starting values to each card
        enriched = []
        for card in flashcards:
            enriched.append({
                "user_id":        data.user_id,
                "subject_id":     data.subject_id,
                "front":          card["front"],
                "back":           card["back"],
                "difficulty":     card["difficulty"],
                "interval_days":  1,        # SM-2 starting interval
                "ease_factor":    2.5,      # SM-2 starting ease factor
                "next_review":    "today",
                "times_reviewed": 0,
                "last_rating":    None
            })

        return {
            "flashcards": enriched,
            "count":      len(enriched)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/flashcards/due")
async def get_due_flashcards(user_id: str, subject_id: str, today: str = None):
    """
    Returns flashcards due for review today.
    Call this before a study session starts.
    The frontend filters by next_review_date <= today from its local database.
    This endpoint is a utility for frontends that store cards server-side.
    """
    # For now return a reminder — actual filtering happens in Supabase
    # Once Supabase is connected this will query the flashcards table
    return {
        "message": "Filter your local flashcards where next_review_date <= today",
        "today": today or datetime.now().strftime("%Y-%m-%d"),
        "user_id": user_id,
        "subject_id": subject_id
    }
class FlashcardRating(BaseModel):
    user_id:     str
    card_id:     str
    rating:      int    # 1=forgot, 2=hard, 3=okay, 4=good, 5=easy

@app.post("/flashcards/rate")
async def rate_flashcard(data: FlashcardRating):
    """
    Update a flashcard's spaced repetition interval after the user reviews it.
    Uses SM-2 algorithm to calculate next review date.
    Call this after the user sees each card and rates how well they remembered it.
    """
    try:
        new_interval, new_ease, next_date = calculate_sm2(
            rating        = data.rating,
            interval_days = 1,    # fetch from DB once Supabase is connected
            ease_factor   = 2.5   # fetch from DB once Supabase is connected
        )

        return {
            "card_id":          data.card_id,
            "new_interval_days": new_interval,
            "new_ease_factor":   new_ease,
            "next_review_date":  next_date,
            "message":           f"Next review scheduled for {next_date}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def calculate_sm2(rating: int, interval_days: int, ease_factor: float) -> tuple:
    """
    SM-2 spaced repetition algorithm.
    rating: 1-5 where 1=complete blackout, 5=perfect recall
    Returns: (new_interval, new_ease_factor, next_review_date)
    """
    # Convert 1-5 rating to SM-2 quality score (0-5)
    quality = rating - 1

    # Update ease factor
    new_ease = ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    new_ease = max(1.3, new_ease)   # minimum ease factor is 1.3

    # Calculate new interval
    if quality < 3:
        # Failed recall — reset to beginning
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
# ── Request model ─────────────────────────────────────────────────────────────
class QARequest(BaseModel):
    user_id:       str
    question:      str
    subject_name:  str
    user_category: str
    context:       Optional[str] = None    # optional study material for grounded answers
    user_profile:  Optional[dict] = None   # optional — for condition-aware responses

@app.post("/qa")
async def study_qa(data: QARequest):
    """
    Answer a study question using Gemini.
    Tailored to the user's category, subject, and learning profile.
    Optionally accepts context (notes or material) to answer from.
    Adapts explanation style for ADHD, dyslexia, autism, and anxiety.
    """
    try:
        from google import genai

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        context_block = (
            f"\nUse this material as your primary reference:\n{data.context[:4000]}"
            if data.context else ""
        )

        # ── Build condition-aware response instructions ────────────────────────
        profile       = data.user_profile or {}
        has_adhd      = profile.get("has_adhd", 0)
        has_dyslexia  = profile.get("has_dyslexia", 0)
        has_autism    = profile.get("has_autism", 0)
        has_anxiety   = profile.get("has_anxiety", 0)

        condition_note = ""

        if has_adhd:
            condition_note = """
            Response style for this student:
            - Use very short paragraphs. Maximum 3 sentences per paragraph.
            - Get to the answer immediately. No long preamble.
            - Use headers or bold key terms to break up the response.
            - Maximum 3 paragraphs total.
            """
        elif has_dyslexia:
            condition_note = """
            Response style for this student:
            - Use simple, everyday words only.
            - Very short sentences. One idea per sentence.
            - Maximum 2 short paragraphs.
            - No complex vocabulary in the explanation itself.
            """
        elif has_autism:
            condition_note = """
            Response style for this student:
            - Be completely literal and explicit. No idioms or metaphors.
            - Structure the answer clearly with a definition first, then explanation.
            - Be precise and factual. No vague language.
            - Maximum 3 paragraphs.
            """
        elif has_anxiety:
            condition_note = """
            Response style for this student:
            - Use a calm, reassuring tone.
            - Frame the explanation as exploration not examination.
            - If the topic is complex, normalise it first before explaining.
            - Maximum 3 short paragraphs.
            """

        prompt = f"""
        You are a knowledgeable study coach helping a {data.user_category} 
        with {data.subject_name}.

        The student asks: "{data.question}"
        {context_block}

        {condition_note}

        Answer clearly and helpfully following these rules:
        - Explain at the right level for a {data.user_category}
        - If this is a concept question, explain the idea first then give an example
        - If this is a calculation or technical question, show the steps clearly
        - If this is a language question, give the answer with an example in context
        - Keep your answer focused — do not pad with unnecessary information
        - If you are not confident about something, say so honestly
        - Maximum 4 short paragraphs unless a longer answer is genuinely needed
        """

        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=prompt
        )

        return {
            "question":     data.question,
            "answer":       response.text.strip(),
            "subject":      data.subject_name,
            "user_id":      data.user_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))