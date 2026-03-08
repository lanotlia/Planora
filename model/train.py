import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
import json
import os
from model.data import TECHNIQUES

def train():
    df = pd.read_csv("model/training_data.csv")
    
    print("String columns found:", df.select_dtypes(include="object").columns.tolist())

    # ── Encode categoricals ──────────────────────────────────────────────────
    categorical_cols = [
        "attention_span", "learning_style", "peak_focus_time",
    "study_env", "user_category", "struggle",
    "content_type", "memory_load", "prior_attempt",
    "current_level"
        # notice: "technique" is gone — it's now the labels not a feature
    ]
    
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # ── Features: everything except the technique columns ────────────────────
    feature_cols = [c for c in df.columns if c not in TECHNIQUES]
    
    # ── Labels: one column per technique ────────────────────────────────────
    X = df[feature_cols]
    Y = df[TECHNIQUES]   # capital Y — this is now a matrix not a single column

    # ── Train one model per technique ────────────────────────────────────────
    # This gives each technique its own decision boundary
    models = {}
    print("\nTraining models...")
    
    for technique in TECHNIQUES:
        y = Y[technique]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight="balanced"
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        f1 = report.get("1", {}).get("f1-score", 0)
        print(f"  {technique:<30} f1={f1:.2f}")
        
        models[technique] = model

    # ── Feature importance across all models ─────────────────────────────────
    importance_summary = {}
    for technique, model in models.items():
        importance_summary[technique] = dict(
            zip(feature_cols, model.feature_importances_)
        )

    # ── Save everything ──────────────────────────────────────────────────────
    os.makedirs("model/artifacts", exist_ok=True)
    joblib.dump(models,   "model/artifacts/models.pkl")
    joblib.dump(encoders, "model/artifacts/encoders.pkl")
    
    with open("model/artifacts/feature_cols.json", "w") as f:
        json.dump(feature_cols, f)
    
    with open("model/artifacts/feature_importance.json", "w") as f:
        json.dump(importance_summary, f)

    print("\nAll models saved to model/artifacts/")

if __name__ == "__main__":
    train()