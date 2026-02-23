import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from src.features import build_features


# -------------------------------
# Load data
# -------------------------------
df = pd.read_csv("data/appointments.csv")
df = build_features(df)

print("Total rows after feature engineering:", len(df))

# -------------------------------
# Feature list (IMPORTANT)
# -------------------------------
FEATURES = [
    "Days_Since_Last",
    "Total_Visits",
    "Avg_Gap",
    "Gap_Std",
    "Visits_Last_14",
    "Visits_Last_30",
    "Visits_Last_60",
    "Overdue",
    "DayOfWeek",
    "Month",
    "IsWeekend",
    "Hour"
]

# -------------------------------
# Time-based split
# -------------------------------
cutoff_date = df["Date"].quantile(0.7)
print("Cutoff date:", cutoff_date)

train = df[df["Date"] <= cutoff_date]
test = df[df["Date"] > cutoff_date]

print("Train size:", len(train))
print("Test size:", len(test))

# -------------------------------
# Model pipeline
# -------------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ))
])

pipeline.fit(train[FEATURES], train["Booked_Next_Week"])

# -------------------------------
# Evaluate
# -------------------------------
probs = pipeline.predict_proba(test[FEATURES])[:, 1]
auc = roc_auc_score(test["Booked_Next_Week"], probs)

print("AUC:", round(auc, 3))

# -------------------------------
# Save model
# -------------------------------
joblib.dump(pipeline, "models/logistic_model.pkl")
print("Model saved.")