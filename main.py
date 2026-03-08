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

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP — train model if artifacts don't exist
# ─────────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
from model.predict import predict
from chatbot.checkin import CheckInSession
import gdown
import os

def download_model_artifacts():
    os.makedirs("model/artifacts", exist_ok=True)
    
    files = {
        "model/artifacts/models.pkl":       "1know7HPwQ_aG6bv6nGeFQPjfXOEFmoIy",
        "model/artifacts/encoders.pkl":     "1rKLQj2WAJT5KCQvUy74mCSlno7u69CCi",
        "model/artifacts/feature_cols.json":"1UXyPYz58CiLsZH9dqzcn"


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
    lifespan=lifespan
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


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status":  "StudyCoach API is running",
        "version": "1.0.0",
        "docs":    "/docs"
    }


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
        import google.generativeai as genai
        import json

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")

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

        response = model.generate_content(prompt)
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


# ─────────────────────────────────────────────────────────────────────────────
# STUDY Q&A ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/qa")
async def study_qa(
    user_id:       str,
    question:      str,
    subject_name:  str,
    user_category: str,
    context:       Optional[str] = None
):
    """
    Answer a study question using Gemini.
    Tailored to the user's category and subject.
    Optionally accepts context (notes or material) to answer from.
    """
    try:
        import google.generativeai as genai

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")

        context_block = (
            f"\nUse this material as your primary reference:\n{context[:4000]}"
            if context else ""
        )

        prompt = f"""
        You are a knowledgeable study coach helping a {user_category} with {subject_name}.

        The student asks: "{question}"
        {context_block}

        Answer clearly and helpfully. Rules:
        - Explain at the right level for a {user_category}
        - If this is a concept question, explain the idea first then give an example
        - If this is a calculation or technical question, show the steps clearly
        - If this is a language question, give the answer with an example in context
        - Keep your answer focused — do not pad it with unnecessary information
        - If you are not confident about something, say so honestly
        - Maximum 4 short paragraphs
        """

        response = model.generate_content(prompt)

        return {
            "question": question,
            "answer":   response.text.strip(),
            "subject":  subject_name
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))