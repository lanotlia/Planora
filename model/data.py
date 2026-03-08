import pandas as pd
import numpy as np
import random

# ── Every technique your app can recommend ──────────────────────────────────
TECHNIQUES = [
    "pomodoro_short",        # 15-min blocks, for ADHD / short attention spans
    "pomodoro_standard",     # 25-min blocks, standard
    "feynman",               # explain it back in simple terms
    "spaced_repetition",     # scheduled review intervals
    "active_recall",         # testing yourself without looking
    "mind_mapping",          # visual concept connection
    "past_papers",           # exam practice under conditions
    "time_blocking",         # dedicated subject blocks in calendar
    "interleaving",          # mixing topics in one session
    "elaborative_interrogation",  # asking WHY something is true
    "audio_notes",           # recording and listening back
    "colour_coding",         # visual organisation for dyslexia
    "chunking",              # breaking content into small labelled pieces
    "worked_examples",       # studying solved problems before attempting
    "dual_coding",           # Doodle concepts, taking them from verbal to visual 

 # ── Language-specific techniques ────────────────────────
    "shadowing",             # listen and repeat native speakers in real time
    "comprehensible_input",  # consuming content slightly above current level
    "spaced_rep_vocab",      # flashcard-based vocab drilling (Anki-style)
    "output_practice",       # forced speaking or writing practice
    "grammar_in_context",    # learning grammar through real sentences not rules
    "immersion_sessions",    # dedicated time consuming only target language

    # ── Certification-specific techniques ───────────────────────────────────
    "timed_practice",        # doing questions under strict exam time conditions
    "topic_prioritisation",  # identifying high-yield topics and focusing there first
    "exam_technique",        # learning how to read, interpret and answer exam questions
    "revision_sprints",      # intensive short bursts covering one topic completely
    "consistency_system",    # habit-based daily study regardless of motivation
]

def generate_profile(force_category=None):
    
    user_category = force_category if force_category else random.choice(
        ["uni_student", "language_learner", "cert_candidate", "self_study"]
    )
    """Generate one realistic user+subject profile"""
    
    # ── Learning disabilities (can have multiple) ────────────────────────────
    has_adhd     = random.random() < 0.20   # 20% prevalence in dataset
    has_dyslexia = random.random() < 0.15
    has_autism   = random.random() < 0.10
    has_anxiety  = random.random() < 0.25 #returns a number between 0 and 1 (takes no arguments) and checks if it's less than the specified percentage to determine if the user has that condition or not.

    # ── Core profile ────────────────────────────────────────────────────────
    attention_span = random.choice(["under_10", "10_20", "20_45", "45_plus"])
    
    # ADHD users are weighted toward shorter attention spans
    if has_adhd:
        attention_span = random.choice(["under_10", "under_10", "10_20", "10_20", "20_45"])

    sleep_hours      = round(random.uniform(4, 9), 1) #returns a decial number between 4 and 9, inclusive
    daily_study_hrs  = round(random.uniform(0.5, 8), 1)
    learning_style   = random.choice(["read_write", "visual", "auditory", "kinesthetic"])
    peak_focus_time  = random.choice(["morning", "afternoon", "evening", "late_night"])
    study_env        = random.choice(["quiet_home", "noisy_home", "library", "cafe", "varies"])
    
    user_category = random.choice([
        "uni_student", "uni_student", "uni_student",  # weighted heavier
        "language_learner",
        "cert_candidate",
        "self_study"
    ])

    # only generated for cert candidates
    prior_attempt = None
    if user_category == "cert_candidate":
        prior_attempt = random.choice([
        "first_time", "first_time", "first_time",  # most are first timers
        "retaking",
        "recertifying"
    ])
     # ── Current level (language learners only) ───────────────────────────────  ← ADD HERE
    current_level = None
    if user_category == "language_learner":
        current_level = random.choice([
            "beginner", "beginner",
            "basic",
            "conversational", "conversational",
            "advanced",
            "near_fluent"
        ])

    # ── Subject profile ──────────────────────────────────────────────────────

    if user_category == "language_learner":
    # Languages are never calculation-heavy
    # They're skill-based with high memory load (vocabulary, grammar rules)
     content_type = random.choice(["practical", "practical", "mixed", "theory"])
     memory_load  = random.choice(["high", "high", "medium"])

    elif user_category == "cert_candidate":
    # Certs vary widely — tech certs are practical, professional certs are mixed
     content_type = random.choice(["theory", "mixed", "mixed", "practical"])
     memory_load  = random.choice(["high", "high", "medium"])

    elif user_category == "uni_student":
    # Uni students can have any content type including calculations
     content_type = random.choice(["theory", "calculation", "mixed", "practical"])
     memory_load  = random.choice(["high", "medium", "low"])

    elif user_category == "self_study":
    # Self-study is usually practical or mixed, rarely pure calculation
     content_type = random.choice(["practical", "mixed", "mixed", "theory"])
     memory_load  = random.choice(["high", "medium", "medium", "low"])

    difficulty    = random.randint(1, 5) #returns a whole number between 1 and 5, inclusive
    days_to_exam = random.choice([None, 7, 14, 30, 60, 90, 180])
    has_deadline  = days_to_exam is not None #Does this variable contain actual data instead of being completely empty
    #if none, then it returns False, otherwise it returns True. 



    # ── Primary struggle (constrained by user category) ──────────────────────────
    if user_category == "language_learner":
     struggle = random.choice([
        "vocab_doesnt_stick",
        "cant_produce",          # understands but can't speak or write it
        "grammar_confusion",
        "no_practice_time",
        "language_anxiety"         
    ])
     
    elif user_category == "cert_candidate":
     struggle = random.choice([
        "running_out_of_time",
        "covering_all_content",
        "exam_technique",
        "staying_consistent",
        "all_of_the_above",
        "all_of_the_above",   # weighted heavier because it's the most common reality
    ])

    else:
     struggle = random.choice([
        "staying_focused", "remembering", "understanding",
        "managing_time", "motivation", "dont_know_how_to_study"
    ])
    

    return {
        # learner features
        # we convert booleans to integers (0 or 1) because many ML models handle numeric data better than categorical text for binary features. This also allows the model to learn the impact of having vs not having each condition.
        "has_adhd":         int(has_adhd),
        "has_dyslexia":     int(has_dyslexia),
        "has_autism":       int(has_autism),
        "has_anxiety":      int(has_anxiety),
        "attention_span":   attention_span,
        "sleep_hours":      sleep_hours,
        "daily_study_hrs":  daily_study_hrs,
        "learning_style":   learning_style,
        "peak_focus_time":  peak_focus_time,
        "study_env":        study_env,
        "user_category":    user_category,
        "struggle":         struggle,
        "current_level":    current_level if current_level else "not_applicable",  # ← this must be here
        "prior_attempt":    prior_attempt if prior_attempt else "not_applicable",
        # subject features
        "content_type":     content_type,
        "memory_load":      memory_load,
        "difficulty":       difficulty,
        "has_deadline":     int(has_deadline),
        "days_to_exam":     days_to_exam if days_to_exam else 999,
        #if days_to_exam is None, it assigns a large number (999) to indicate no imminent deadline. This allows the model to learn that techniques suited for urgent deadlines should not be recommended when days_to_exam is high.
    }

# assigns technique scores based on the profile using a set of rules derived from learning science research. Each rule adds or subtracts points from techniques based on how well they match the user's needs and challenges. The techniques with the highest scores are then recommended as the best fit for that user+subject profile.
def apply_rules(profile: dict) -> list:
    """
    Takes a profile dict, returns a ranked list of recommended techniques.
    This function encodes learning science research directly.
    Every decision here should be citeable in your dissertation.
    """
    scores = {t: 0 for t in TECHNIQUES}

    # ── CONTENT TYPE RULES ───────────────────────────────────────────────────
    if profile["content_type"] == "theory":
        scores["feynman"]                  += 3
        scores["spaced_repetition"]        += 3
        scores["elaborative_interrogation"]+= 2
        scores["active_recall"]            += 2
        scores["mind_mapping"]             += 1

    elif profile["content_type"] == "calculation":
        scores["past_papers"]              += 4
        scores["worked_examples"]          += 3
        scores["spaced_repetition"]        += 2
        scores["active_recall"]            += 2
        scores["time_blocking"]            += 1

    elif profile["content_type"] == "mixed":
        scores["time_blocking"]            += 4  # separate theory vs calc blocks
        scores["feynman"]                  += 2
        scores["past_papers"]              += 2
        scores["spaced_repetition"]        += 2
        scores["active_recall"]            += 2

    elif profile["content_type"] == "practical":
        scores["chunking"]                 += 3
        scores["worked_examples"]          += 3
        scores["active_recall"]            += 2
        scores["spaced_repetition"]        += 1

    # ── ADHD RULES ───────────────────────────────────────────────────────────
    if profile["has_adhd"]:
        scores["pomodoro_short"]           += 4
        scores["pomodoro_standard"]        -= 2  # too long for ADHD
        scores["time_blocking"]            += 3
        scores["interleaving"]             += 2  # variety prevents boredom
        scores["chunking"]                 += 3
        scores["feynman"]                  += 1  # active engagement helps ADHD
        
        # ADHD + noisy environment is a bad combo, boost structure
        if profile["study_env"] in ["noisy_home", "varies"]:
            scores["time_blocking"]        += 2
            scores["chunking"]             += 1

    else:
        scores["pomodoro_standard"]        += 2

    # ── DYSLEXIA RULES ───────────────────────────────────────────────────────
    if profile["has_dyslexia"]:
        scores["audio_notes"]              += 4
        scores["colour_coding"]            += 3
        scores["mind_mapping"]             += 3  # visual over text
        scores["chunking"]                 += 2
        scores["feynman"]                  += 2  # verbal explanation over reading
        scores["active_recall"]            -= 1  # if text-based, harder for dyslexia

    # ── AUTISM RULES ─────────────────────────────────────────────────────────
    if profile["has_autism"]:
        scores["time_blocking"]            += 4  # predictable structure
        scores["spaced_repetition"]        += 2  # consistent routine
        scores["chunking"]                 += 2  # clear defined pieces
        scores["interleaving"]             -= 3  # topic switching is stressful

    # ── ANXIETY RULES ────────────────────────────────────────────────────────
    if profile["has_anxiety"]:
        scores["chunking"]                 += 3  # smaller = less overwhelming
        scores["pomodoro_short"]           += 2  # short wins reduce anxiety
        scores["time_blocking"]            += 2  # knowing the plan reduces anxiety
        scores["past_papers"]              += 1  # familiarity with exam format helps
        scores["elaborative_interrogation"]-= 1  # can feel overwhelming when anxious

    # ── ATTENTION SPAN RULES ─────────────────────────────────────────────────
    if profile["attention_span"] == "under_10":
        scores["pomodoro_short"]           += 4
        scores["chunking"]                 += 3
        scores["interleaving"]             += 2
        scores["pomodoro_standard"]        -= 3

    elif profile["attention_span"] == "10_20":
        scores["pomodoro_short"]           += 2
        scores["chunking"]                 += 2

    elif profile["attention_span"] in ["20_45", "45_plus"]:
        scores["pomodoro_standard"]        += 2
        scores["feynman"]                  += 1
        scores["elaborative_interrogation"]+= 1

    # ── LEARNING STYLE RULES ─────────────────────────────────────────────────
    if profile["learning_style"] == "visual":
        scores["mind_mapping"]             += 3
        scores["colour_coding"]            += 2
        scores["time_blocking"]            += 1

    elif profile["learning_style"] == "auditory":
        scores["audio_notes"]              += 3
        scores["feynman"]                  += 2  # talking through material

    elif profile["learning_style"] == "kinesthetic":
        scores["past_papers"]              += 2
        scores["worked_examples"]          += 2
        scores["active_recall"]            += 2

    elif profile["learning_style"] == "read_write":
        scores["active_recall"]            += 2
        scores["elaborative_interrogation"]+= 2
        scores["spaced_repetition"]        += 1

    # ── DUAL CODING RULES ────────────────────────────────────────────────────
    # Core triggers — these profiles benefit most from dual coding
    if profile["learning_style"] == "visual":
        scores["dual_coding"]              += 4

    if profile["content_type"] in ["theory", "mixed"]:
        scores["dual_coding"]              += 2  # abstract content benefits most

    if profile["memory_load"] == "high":
        scores["dual_coding"]              += 2  # two channels aid heavy memorisation

    if profile["has_dyslexia"]:
        scores["dual_coding"]              += 3  # reduces reliance on text alone

    if profile["has_autism"]:
        scores["dual_coding"]              += 2  # visual structure aids processing

    if profile["struggle"] in ["remembering", "understanding"]:
        scores["dual_coding"]              += 2

    # Works well alongside these techniques
    if profile["learning_style"] == "auditory":
        scores["dual_coding"]              -= 1  # auditory learners benefit less

    if profile["content_type"] == "calculation":
        scores["dual_coding"]              -= 1  # less relevant for pure calculation

# ── COLOUR CODING RULES ──────────────────────────────────────────────────
    # standalone block — paste exactly here
    if profile["has_dyslexia"]:
        scores["colour_coding"]            += 4
    if profile["learning_style"] == "visual":
        scores["colour_coding"]            += 3
    if profile["memory_load"] == "high":
        scores["colour_coding"]            += 2
    if profile["content_type"] == "theory":
        scores["colour_coding"]            += 1
    if profile["struggle"] == "remembering":
        scores["colour_coding"]            += 2

# ── INTERLEAVING RULES ───────────────────────────────────────────────────
    if profile["has_adhd"]:
        scores["interleaving"]             += 3  # variety prevents boredom

    if profile["user_category"] == "uni_student":
        scores["interleaving"]             += 2  # multiple subjects benefit from mixing

    if profile["struggle"] == "staying_focused":
        scores["interleaving"]             += 2  # topic switches re-engage attention

    if profile["struggle"] == "motivation":
        scores["interleaving"]             += 2  # variety maintains interest

    if profile["days_to_exam"] > 60 or not profile["has_deadline"]:
        scores["interleaving"]             += 2  # long-term retention beats blocked practice

    if profile["has_autism"]:
        scores["interleaving"]             -= 3  # topic switching is stressful for autism

# ── REVISION SPRINTS RULES ───────────────────────────────────────────────
    if profile["has_deadline"] and profile["days_to_exam"] <= 30:
        scores["revision_sprints"]         += 4
    
    if profile["has_deadline"] and profile["days_to_exam"] <= 14:
        scores["revision_sprints"]         += 3  # stacks with above for urgent cases

    if profile["user_category"] == "cert_candidate":
        scores["revision_sprints"]         += 2  # all cert candidates benefit

    if profile["memory_load"] == "high" and profile["has_deadline"]:
        scores["revision_sprints"]         += 2

    if profile["struggle"] in ["covering_all_content", "managing_time"]:
        scores["revision_sprints"]         += 2

# ── CONSISTENCY SYSTEM RULES ─────────────────────────────────────────────
    if profile["user_category"] in ["self_study", "language_learner"]:
        scores["consistency_system"]       += 4  # no external deadlines = needs habit system

    if profile["user_category"] == "cert_candidate":
        scores["consistency_system"]       += 3

    if profile["struggle"] == "motivation":
        scores["consistency_system"]       += 4

    if profile["struggle"] == "staying_consistent":
        scores["consistency_system"]       += 5

    if profile["struggle"] == "no_practice_time":
        scores["consistency_system"]       += 3

    if not profile["has_deadline"]:
        scores["consistency_system"]       += 2  # no deadline = needs self-imposed structure

# ── EXAM TECHNIQUE RULES ─────────────────────────────────────────────────
    if profile["has_deadline"] and profile["days_to_exam"] <= 60:
        scores["exam_technique"]           += 3

    if profile["user_category"] == "cert_candidate":
        scores["exam_technique"]           += 3

    if profile["struggle"] in ["exam_technique", "running_out_of_time"]:
        scores["exam_technique"]           += 4

    if profile.get("prior_attempt") == "retaking":
        scores["exam_technique"]           += 3  # retakers often failed on technique

    if profile["content_type"] == "mixed":
        scores["exam_technique"]           += 1  # mixed exams need strategic approach

# ── TIMED PRACTICE RULES — update existing block ─────────────────────────
    if profile["has_deadline"] and profile["days_to_exam"] <= 60:
        scores["timed_practice"]           += 3
    if profile["has_deadline"] and profile["days_to_exam"] <= 30:
        scores["timed_practice"]           += 2  # stacks for urgency
    if profile["user_category"] == "cert_candidate":
        scores["timed_practice"]           += 3  # was 2, increase baseline
    if profile["user_category"] == "uni_student" and profile["has_deadline"]:
        scores["timed_practice"]           += 2  # uni students with exams need this too
    if profile["struggle"] in ["exam_technique", "running_out_of_time"]:
        scores["timed_practice"]           += 3
    if profile.get("prior_attempt") == "retaking":
        scores["timed_practice"]           += 2

    # ── TOPIC PRIORITISATION RULES — update existing block ───────────────────
    if profile["user_category"] == "cert_candidate":
        scores["topic_prioritisation"]     += 3  # was 2, increase baseline
    if profile["user_category"] == "uni_student" and profile["has_deadline"]:
        scores["topic_prioritisation"]     += 2  # exam prep needs prioritisation
    if profile["has_deadline"] and profile["days_to_exam"] <= 30:
        scores["topic_prioritisation"]     += 3  # running out of time = must prioritise
    if profile["memory_load"] == "high" and profile["has_deadline"]:
        scores["topic_prioritisation"]     += 2
    if profile["struggle"] == "covering_all_content":
        scores["topic_prioritisation"]     += 3
    if profile.get("prior_attempt") == "retaking":
        scores["topic_prioritisation"]     += 2


# ── DEADLINE PRESSURE RULES ──────────────────────────────────────────────
    if profile["has_deadline"]:
        if profile["days_to_exam"] <= 14:    # exam very soon
            scores["past_papers"]          += 4
            scores["active_recall"]        += 3
            scores["spaced_repetition"]    += 2
            scores["feynman"]              -= 1  # not enough time for deep learning
            
        elif profile["days_to_exam"] <= 30:
            scores["past_papers"]          += 2
            scores["spaced_repetition"]    += 3
            scores["active_recall"]        += 2

        elif profile["days_to_exam"] <= 90:
            scores["spaced_repetition"]    += 3
            scores["feynman"]              += 2
            scores["time_blocking"]        += 2

    # ── SLEEP DEPRIVATION RULES ──────────────────────────────────────────────
    if profile["sleep_hours"] < 6:
        scores["spaced_repetition"]        += 2  # more frequent shorter reviews
        scores["chunking"]                 += 2  # smaller sessions
        scores["pomodoro_short"]           += 1
        scores["feynman"]                  -= 1  # deep processing needs good sleep

    # ── MEMORY LOAD RULES ────────────────────────────────────────────────────
    if profile["memory_load"] == "high":
        scores["spaced_repetition"]        += 3
        scores["active_recall"]            += 3
        scores["chunking"]                 += 1

    elif profile["memory_load"] == "low":
        scores["feynman"]                  += 2
        scores["elaborative_interrogation"]+= 2
        scores["mind_mapping"]             += 1

    # ── STRUGGLE RULES ───────────────────────────────────────────────────────
    if profile["struggle"] == "staying_focused":
        scores["pomodoro_short"]           += 2
        scores["time_blocking"]            += 2
        scores["interleaving"]             += 1

    elif profile["struggle"] == "remembering":
        scores["spaced_repetition"]        += 3
        scores["active_recall"]            += 2

    elif profile["struggle"] == "understanding":
        scores["feynman"]                  += 3
        scores["elaborative_interrogation"]+= 2

    elif profile["struggle"] == "dont_know_how_to_study":
        scores["feynman"]                  += 2
        scores["time_blocking"]            += 2
        scores["spaced_repetition"]        += 2
        scores["chunking"]                 += 1
    
# ── LANGUAGE LEARNER STRUGGLE RULES ──────────────────────────────────────
    # ── LANGUAGE LEARNER RULES ───────────────────────────────────────────────
    if profile["user_category"] == "language_learner":
        
        # These apply to ALL language learners regardless of struggle
        scores["spaced_rep_vocab"]         += 4  # every language learner needs vocab drilling
        scores["comprehensible_input"]     += 3  # every language learner benefits from input
        scores["immersion_sessions"]       += 3  # baseline recommendation for all
        scores["grammar_in_context"]       += 3
        scores["output_practice"]          += 2
        scores["consistency_system"]       += 4
        scores["shadowing"]                += 2
        
         # also add level-based triggers
        if profile.get("current_level") in ["beginner", "basic"]:
            scores["spaced_rep_vocab"]     += 3  # vocabulary is everything at early stages
            scores["grammar_in_context"]   += 2
            scores["comprehensible_input"] += 2

        if profile.get("current_level") in ["conversational", "advanced", "near_fluent"]:
            scores["output_practice"]      += 2
            scores["immersion_sessions"]   += 2
            scores["spaced_rep_vocab"]     += 2  # vocab never stops being important

        # Auditory learners get shadowing as a baseline too
        if profile["learning_style"] == "auditory":
            scores["shadowing"]            += 3
        
        
        # Then struggle-specific on top
        if profile["struggle"] == "vocab_doesnt_stick":
            scores["spaced_rep_vocab"]         += 4
            scores["comprehensible_input"]     += 2
            scores["immersion_sessions"]       += 1
            scores["active_recall"]            += 2
            scores["chunking"]                 += 1

        elif profile["struggle"] == "cant_produce":
            scores["output_practice"]          += 5
            scores["shadowing"]                += 4
            scores["feynman"]                  += 2
            scores["comprehensible_input"]     += 1

        elif profile["struggle"] == "grammar_confusion":
            scores["grammar_in_context"]       += 4
            scores["comprehensible_input"]     += 3
            scores["elaborative_interrogation"]+= 2
            scores["active_recall"]            += 2
            scores["spaced_rep_vocab"]         += 2

        elif profile["struggle"] == "no_practice_time":
            scores["immersion_sessions"]       += 4
            scores["shadowing"]                += 3
            scores["pomodoro_short"]           += 3
            scores["spaced_rep_vocab"]         += 2
            scores["time_blocking"]            += 2

        elif profile["struggle"] == "language_anxiety":
            scores["output_practice"]          += 3
            scores["comprehensible_input"]     += 3
            scores["shadowing"]                += 2
            scores["chunking"]                 += 2
            scores["pomodoro_short"]           += 2
            scores["past_papers"]              -= 2
            scores["active_recall"]            -= 1

       

    # ── CERT CANDIDATE STRUGGLE RULES ────────────────────────────────────────
   # ── CERT CANDIDATE RULES ─────────────────────────────────────────────────
    if profile["user_category"] == "cert_candidate":
        
        # These apply to ALL cert candidates regardless of struggle
        scores["past_papers"]              += 3  # universal for cert prep
        scores["timed_practice"]           += 2  # every cert has time pressure
        scores["topic_prioritisation"]     += 2  # every cert has a syllabus to prioritise
        scores["spaced_repetition"]        += 2
        
        # Struggle-specific on top
        if profile["struggle"] == "running_out_of_time":
            scores["timed_practice"]           += 3
            scores["exam_technique"]           += 4
            scores["past_papers"]              += 2
            scores["time_blocking"]            += 2
            scores["chunking"]                 += 1

        elif profile["struggle"] == "covering_all_content":
            scores["topic_prioritisation"]     += 3
            scores["time_blocking"]            += 4
            scores["spaced_repetition"]        += 2
            scores["revision_sprints"]         += 3
            scores["chunking"]                 += 2

        elif profile["struggle"] == "exam_technique":
            scores["exam_technique"]           += 5
            scores["timed_practice"]           += 3
            scores["past_papers"]              += 2
            scores["active_recall"]            += 2
            scores["worked_examples"]          += 1

        elif profile["struggle"] == "staying_consistent":
            scores["consistency_system"]       += 5
            scores["time_blocking"]            += 4
            scores["pomodoro_short"]           += 3
            scores["spaced_repetition"]        += 2
            scores["chunking"]                 += 2

        elif profile["struggle"] == "all_of_the_above":
            scores["timed_practice"]           += 3
            scores["topic_prioritisation"]     += 3
            scores["time_blocking"]            += 3
            scores["consistency_system"]       += 3
            scores["past_papers"]              += 2
            scores["spaced_repetition"]        += 2
            scores["exam_technique"]           += 2
            scores["revision_sprints"]         += 2
            scores["chunking"]                 += 1
    # ── CERT RETAKER RULES ───────────────────────────────────────────────────
    if profile.get("prior_attempt") == "retaking":
        scores["topic_prioritisation"]     += 4  # they have gaps not full ignorance
        scores["exam_technique"]           += 3  # likely failed on technique not knowledge
        scores["timed_practice"]           += 3
        scores["past_papers"]              += 2
        scores["revision_sprints"]         += 2
        # deprioritise full coverage techniques — they don't need to start from scratch
        scores["spaced_repetition"]        -= 1
        scores["consistency_system"]       -= 1


    # ── RANK AND RETURN TOP 5 ────────────────────────────────────────────────
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_5  = [t for t, s in ranked if s > 0][:5]
    
    return top_5


def generate_dataset(n_per_category=400):
    """Generate balanced dataset with equal representation per user category"""
    rows = []
    categories = ["uni_student", "language_learner", "cert_candidate", "self_study"]
    
    for category in categories:
        for _ in range(n_per_category):
            profile = generate_profile(force_category=category)
            techniques = apply_rules(profile)
            
            row = profile.copy()
            for technique in TECHNIQUES:
                row[technique] = 1 if technique in techniques else 0
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv("model/training_data.csv", index=False)
    total = len(df)
    print(f"Generated {total} rows ({n_per_category} per category × {len(categories)} categories)")
    print(f"\\nSample technique distribution:\\n{df[TECHNIQUES].sum().sort_values(ascending=False)}")
    return df


if __name__ == "__main__":
    generate_dataset(1000)


