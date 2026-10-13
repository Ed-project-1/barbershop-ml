import pandas as pd
import joblib
from pathlib import Path
from src.features import FEATURES, build_features


# -------------------------------
# Load data
# -------------------------------
def main():
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
    artifact = joblib.load("models/logistic_model.pkl")
    if not isinstance(artifact, dict):
        raise ValueError("Retrain using python -m src.train --horizon 31 (or another reviewed window).")
    model = artifact["pipeline"]
    days = artifact["horizon_days"]
    print(f"Scoring returns within {days} days after each latest recorded appointment.")
    latest["Prediction_Window_Days"] = days
    latest["Window_End_Date"] = latest["Date"] + pd.Timedelta(days=days)

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
    Path("campaign_logs").mkdir(parents=True, exist_ok=True)
    active_campaign.to_csv("campaign_logs/active_campaign.csv", index=False)
    reactivation_campaign.to_csv("campaign_logs/reactivation_campaign.csv", index=False)

    print("Campaign files saved.")


if __name__ == "__main__":
    main()
