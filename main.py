from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from model.predict import predict
from chatbot.checkin import CheckInSession
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class UserProfile(BaseModel):
    has_adhd:        int
    has_dyslexia:    int
    has_autism:      int
    has_anxiety:     int
    attention_span:  str
    sleep_hours:     float
    daily_study_hrs: float
    learning_style:  str
    peak_focus_time: str
    study_env:       str
    user_category:   str
    struggle:        str

class SubjectProfile(BaseModel):
    subject_name:  str
    content_type:  str
    memory_load:   str
    difficulty:    int
    has_deadline:  int
    days_to_exam:  Optional[int] = 999

# Store active sessions in memory
# In production this would be Redis, for demo this is fine
active_sessions: dict = {}

class CheckInStart(BaseModel):
    user_id:      str
    user_profile: dict
    subject_name: str
    material:     Optional[str] = None   # pasted text or extracted PDF text

class CheckInMessage(BaseModel):
    user_id:  str
    message:  str

@app.post("/checkin/start")
async def start_checkin(data: CheckInStart):
    """Start a new check-in session"""
    session = CheckInSession(
        user_profile = data.user_profile,
        subject_name = data.subject_name,
        material     = data.material
    )
    opening_message = session.start()
    
    # Store session keyed by user_id
    active_sessions[data.user_id] = session

    return {
        "message":  opening_message,
        "stage":    "checkin",
        "user_id":  data.user_id
    }

@app.post("/checkin/message")
async def send_checkin_message(data: CheckInMessage):
    """Handle each message in the check-in conversation"""
    
    session = active_sessions.get(data.user_id)
    if not session:
        return {"error": "No active session found. Start a new check-in first."}

    # Route message to correct handler based on current stage
    if session.stage == "checkin":
        response = session.process_checkin_response(data.message)
        return {
            "message": response,
            "stage":   session.stage,
        }

    elif session.stage == "quiz":
        result = session.process_quiz_answer(data.message)
        
        # process_quiz_answer returns tuple when quiz is complete
        if isinstance(result, tuple):
            message, summary = result
            return {
                "message":  message,
                "stage":    "complete",
                "summary":  summary,
                "session":  session.get_session_data()
            }
        
        return {
            "message": result,
            "stage":   "quiz",
        }

    else:
        return {
            "message": "Session complete.",
            "stage":   "complete"
        }

@app.post("/recommend")
async def get_recommendations(user: UserProfile, subject: SubjectProfile):
    result = predict(user.dict(), subject.dict())
    return result

@app.get("/")
async def root():
    return {"status": "StudyCoach API is running"}