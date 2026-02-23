import pandas as pd
import joblib
from src.features import build_features


# -------------------------------
# Load data
# -------------------------------
df = pd.read_csv("data/appointments.csv")
df = build_features(df)

# Latest record per phone
latest = (
    df.sort_values("Date")
    .groupby("CustomerPhone")
    .tail(1)
    .copy()
)

# -------------------------------
# Load model
# -------------------------------
model = joblib.load("models/logistic_model.pkl")

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

latest["Booking_Probability"] = model.predict_proba(
    latest[FEATURES]
)[:, 1]

# -------------------------------
# Segmentation
# -------------------------------
today = pd.Timestamp.today().normalize()

latest["Days_Since_Last_Visit"] = (
    today - latest["Date"]
).dt.days

threshold = 0.6
active_limit = 120
reactivation_limit = 420

active_campaign = latest[
    (latest["Booking_Probability"] >= threshold) &
    (latest["Days_Since_Last_Visit"] <= active_limit)
]

reactivation_campaign = latest[
    (latest["Days_Since_Last_Visit"] > active_limit) &
    (latest["Days_Since_Last_Visit"] <= reactivation_limit)
]

# Enforce exclusivity
reactivation_campaign = reactivation_campaign[
    ~reactivation_campaign["CustomerPhone"].isin(
        active_campaign["CustomerPhone"]
    )
]

print("Active:", len(active_campaign))
print("Reactivation:", len(reactivation_campaign))

# Save output
active_campaign.to_csv("campaign_logs/active_campaign.csv", index=False)
reactivation_campaign.to_csv("campaign_logs/reactivation_campaign.csv", index=False)

print("Campaign files saved.")