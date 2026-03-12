import joblib
import json
import os
from model.data import TECHNIQUES, apply_rules

# ── Lazy loading — models load on first prediction, not at import ─────────────
_models   = None
_encoders = None
_feature_cols = None

def _load_artifacts():
    global _models, _encoders, _feature_cols
    
    if _models is None:
        print("Loading model artifacts...")
        _models   = joblib.load("model/artifacts/models.pkl")
        _encoders = joblib.load("model/artifacts/encoders.pkl")
        
        with open("model/artifacts/feature_cols.json") as f:
            _feature_cols = json.load(f)
        print("Model artifacts loaded.")
    
    return _models, _encoders, _feature_cols

# ── Technique descriptions for the frontend ──────────────────────────────────
TECHNIQUE_INFO = {
    "dual_coding": {
        "name": "dual_coding",
        "how_to": 
            "Every time you write notes on a concept, draw a visual representation "
            "alongside it — a diagram, timeline, flowchart, or sketch. They must "
            "represent the same information in two different forms. Do not just "
            "decorate your notes; the visual must carry the same meaning as the text. "
            "When reviewing, cover the text and try to recall it from the visual alone, "
            "then cover the visual and recall from the text."
        ,
        "why": 
            "Based on Paivio's Dual Coding Theory — the brain processes verbal and "
            "visual information through separate channels. Encoding the same material "
            "through both channels creates two retrieval pathways, making recall "
            "significantly more reliable than text alone."
        ,
    },
    "pomodoro_short":           {
        "name": "Short Pomodoro (15 min)",
        "how_to": "Set a timer for 15 minutes. Study only that subject. When it rings, take a 5-minute break. Repeat.",
        "why":    "Short focused bursts prevent mental fatigue and work with your natural attention rhythm."
    },
    "pomodoro_standard":        {
        "name": "Pomodoro Technique (25 min)",
        "how_to": "25 minutes of focused study, 5-minute break. After 4 rounds, take a 20-minute break.",
        "why":    "Creates sustainable focus sessions with built-in recovery."
    },
    "feynman":                  {
        "name": "Feynman Technique",
        "how_to": "After studying a concept, close your notes and explain it out loud as if teaching a 10-year-old. Where you struggle to explain is where your understanding has gaps.",
        "why":    "Forces deep understanding over surface memorisation."
    },
    "spaced_repetition":        {
        "name": "Spaced Repetition",
        "how_to": "Review material at increasing intervals: 1 day, 3 days, 7 days, 14 days, 30 days. The app schedules this for you automatically.",
        "why":    "Exploits the spacing effect — the most proven memory technique in cognitive science."
    },
    "active_recall":            {
        "name": "Active Recall",
        "how_to": "Instead of re-reading notes, close them and write down or say everything you can remember. Then check what you missed.",
        "why":    "Testing yourself is significantly more effective than passive review."
    },
    "mind_mapping":             {
        "name": "Mind Mapping",
        "how_to": "Put the main concept in the centre of a page. Branch out with related ideas, subtopics, and connections. Use colour and images.",
        "why":    "Visual organisation helps you see relationships between concepts and suits visual learners."
    },
    "past_papers":              {
        "name": "Past Paper Practice",
        "how_to": "Get past exam papers or practice questions. Do them under timed exam conditions. Mark your answers and analyse every mistake.",
        "why":    "The most direct preparation for exams. Builds familiarity with question formats and reveals real gaps."
    },
    "time_blocking":            {
        "name": "Time Blocking",
        "how_to": "Assign specific subjects to specific time slots in your calendar before the day starts. Treat each block like an appointment you cannot miss.",
        "why":    "Removes daily decision fatigue and ensures every subject gets dedicated attention."
    },
    "interleaving":             {
        "name": "Interleaved Practice",
        "how_to": "Instead of studying one subject for hours, rotate between 2-3 topics in a session. Spend 20-30 minutes on each before switching.",
        "why":    "Switching topics feels harder but produces stronger long-term retention and keeps engagement high."
    },
    "elaborative_interrogation": {
        "name": "Elaborative Interrogation",
        "how_to": "For every fact or concept, ask yourself WHY it is true and HOW it connects to things you already know. Write down your answers.",
        "why":    "Connecting new information to existing knowledge dramatically improves retention."
    },
    "audio_notes":              {
        "name": "Audio Notes",
        "how_to": "Record yourself summarising your notes out loud. Listen back during walks, commutes, or before sleep.",
        "why":    "Suits auditory learners and adds a second encoding pathway beyond reading."
    },
    "colour_coding":            {
        "name": "Colour Coding",
        "how_to": "Use consistent colours across all your notes — e.g. blue for definitions, yellow for examples, red for things to memorise. Be consistent.",
        "why":    "Visual organisation reduces cognitive load and helps with text processing difficulties."
    },
    "chunking":                 {
        "name": "Chunking",
        "how_to": "Break large topics into small named pieces. Study one chunk completely before moving on. Give each chunk a clear label.",
        "why":    "Prevents overwhelm and makes large syllabuses feel manageable."
    },
    "worked_examples":          {
        "name": "Worked Examples",
        "how_to": "Before attempting problems yourself, study fully solved examples step by step. Understand every line before moving on.",
        "why":    "Reduces cognitive load when learning new problem types and builds correct mental models."
    },
    "shadowing": {
        "name": "Shadowing",
        "how_to": "Find a native speaker audio clip with a transcript. Play it, and speak along simultaneously, matching their rhythm, tone and pronunciation exactly. Start slow, then build up to natural speed.",
        "why":   "Directly trains your brain to produce the language, not just recognise it. Closes the gap between understanding and speaking."
    },
    "comprehensible_input": {
        "name": "Comprehensible Input",
        "how_to": "Consume content in your target language that is just slightly above your level — podcasts, YouTube, books, films with subtitles. You should understand about 70-80% without effort.",
        "why":   "Based on Krashen's Input Hypothesis — language is acquired naturally when you receive meaningful input at the right level, not through drilling rules."
    },
    "spaced_rep_vocab": {
        "name": "Spaced Repetition Vocabulary",
        "how_to": "Add new words to your flashcard deck daily. The app schedules reviews automatically — review cards every day but only the ones due. Never skip a day even if it's just 5 minutes.",
        "why":   "The forgetting curve means vocabulary fades fast without review. Spaced repetition is the most research-backed method for long-term vocabulary retention."
    },
    "output_practice": {
        "name": "Forced Output Practice",
        "how_to": "Set aside time to only produce the language — speak, write, or type in your target language. Keep a daily journal in the language, find a conversation partner, or use language exchange apps.",
        "why":   "Understanding input and producing output use different cognitive pathways. You must practise production separately — comprehension does not automatically transfer to speaking."
    },
    "grammar_in_context": {
        "name": "Grammar in Context",
        "how_to": "Instead of memorising grammar rules in isolation, find 10 real example sentences using that grammar pattern. Read them, copy them, modify them. Learn the pattern through examples not explanations.",
        "why":   "The brain acquires grammar through pattern recognition in meaningful context far more effectively than abstract rule memorisation."
    },
    "immersion_sessions": {
        "name": "Immersion Sessions",
        "how_to": "Dedicate a time block where you only consume your target language — change your phone language, watch a show without English subtitles, listen to a podcast during your commute. Start with 30 minutes daily.",
        "why":   "Passive immersion builds listening comprehension and natural pattern recognition without requiring active study time."
    },
    "timed_practice": {
        "name": "Timed Practice",
        "how_to": "Do past questions under strict exam conditions — same time limit, no notes, no interruptions. After each attempt, review every wrong answer and understand exactly why it was wrong before moving on.",
        "why":   "Time pressure in exams is a separate skill from knowing the content. You must train under conditions that match the real thing."
    },
    "topic_prioritisation": {
        "name": "Topic Prioritisation",
        "how_to": "Get your exam syllabus or past paper breakdown. Identify which topics appear most frequently and carry the most marks. Study those first and deepest. Cover lower-yield topics last.",
        "why":   "Not all content carries equal weight. Strategic focus on high-yield areas maximises your score per hour of study."
    },
    "exam_technique": {
        "name": "Exam Technique Training",
        "how_to": "Study how your specific exam works — how questions are worded, common traps, marking schemes, time per question. Practise reading questions carefully and identifying exactly what is being asked before answering.",
        "why":   "Many candidates fail not from lack of knowledge but from poor exam strategy. Understanding the format is a learnable skill."
    },
    "revision_sprints": {
        "name": "Revision Sprints",
        "how_to": "Pick one topic. Study it completely in one focused session — notes, examples, practice questions. Do not move to the next topic until this one is done. Aim to fully close each topic before opening the next.",
        "why":   "Prevents the common trap of partially covering everything and fully mastering nothing."
    },
    "consistency_system": {
        "name": "Consistency System",
        "how_to": "Attach studying to an existing daily habit — after breakfast, before bed, during your lunch break. Start with just 20 minutes daily. Track your streak. Missing one day is allowed, never miss two in a row.",
        "why":   "Motivation fluctuates but systems are reliable. Small consistent effort over weeks beats large irregular sessions every time."
    },
}

def _encode_profile(combined: dict) -> list:
    models, encoders, feature_cols = _load_artifacts()  # ← gets feature_cols here
    
    categorical_cols = [
        "attention_span", "learning_style", "peak_focus_time",
        "study_env", "user_category", "struggle",
        "content_type", "memory_load", "prior_attempt",
        "current_level"
    ]
    
    encoded = {}
    for col in feature_cols:          # ← uses local feature_cols
        val = combined.get(col, 0)
        if col in categorical_cols and col in encoders:
            try:
                val = encoders[col].transform([val])[0]
            except ValueError:
                val = 0
        if isinstance(val, str):
            val = 0
        encoded[col] = val
    
    return [encoded[col] for col in feature_cols]   # ← uses local feature_cols


def predict(user_profile: dict, subject_profile: dict) -> dict:
    models, encoders, feature_cols = _load_artifacts()  # ← gets feature_cols here
    combined = {**user_profile, **subject_profile}
    
    rule_based     = apply_rules(combined)
    feature_vector = _encode_profile(combined)
    
    model_scores = {}
    for technique, model in models.items():
        prob = model.predict_proba([feature_vector])[0]
        model_scores[technique] = prob[1] if len(prob) > 1 else 0

    final_techniques = []
    for tech in rule_based:
        confidence = model_scores.get(tech, 0)
        if confidence > 0.3:
            final_techniques.append((tech, confidence))

    final_techniques.sort(key=lambda x: x[1], reverse=True)
    top_5 = final_techniques[:5]

    if len(top_5) < 3:
        top_5 = [(t, 1.0) for t in rule_based[:5]]

    recommendations = []
    for i, (tech, confidence) in enumerate(top_5):
        info = TECHNIQUE_INFO.get(tech, {})
        recommendations.append({
            "rank":       i + 1,
            "id":         tech,
            "name":       info.get("name", tech),
            "how_to":     info.get("how_to", ""),
            "why":        info.get("why", ""),
            "confidence": round(confidence, 2)
        })

    return {
        "subject":         subject_profile.get("subject_name", "Your subject"),
        "recommendations": recommendations,
        "session_length":  _recommend_session_length(combined),
        "daily_sessions":  _recommend_session_count(combined),
    }

def _recommend_session_length(profile: dict) -> str:
    if profile.get("has_adhd") or profile.get("attention_span") == "under_10":
        return "15 minutes per session"
    elif profile.get("attention_span") == "10_20":
        return "20 minutes per session"
    elif profile.get("attention_span") == "20_45":
        return "25 minutes per session"
    else:
        return "45 minutes per session"


def _recommend_session_count(profile: dict) -> str:
    hours = profile.get("daily_study_hrs", 2)
    if hours <= 1:
        return "1–2 sessions per day"
    elif hours <= 3:
        return "3–4 sessions per day"
    else:
        return "5–6 sessions per day"