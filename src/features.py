import pandas as pd
import numpy as np


def build_features(df):

    # -------------------------------
    # Cleaning
    # -------------------------------
    df["Customer"] = df["Customer"].astype(str).str.strip().str.title()
    df["CustomerPhone"] = df["CustomerPhone"].astype(str).str.strip()

    df = df[df["CustomerPhone"].notna()]
    df = df[df["CustomerPhone"] != ""]
    df = df[df["CustomerPhone"] != "nan"]

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Time"] = pd.to_datetime(df["Time"], format="%I:%M %p", errors="coerce")

    df = df.sort_values(["CustomerPhone", "Date"]).copy()

    # -------------------------------
    # Recency
    # -------------------------------
    df["Days_Since_Last"] = (
        df.groupby("CustomerPhone")["Date"]
        .diff()
        .dt.days
    )

    # -------------------------------
    # Frequency
    # -------------------------------
    df["Total_Visits"] = df.groupby("CustomerPhone").cumcount() + 1
    df["Lifetime_Visits"] = df.groupby("CustomerPhone")["CustomerPhone"].transform("count")

    # -------------------------------
    # Average return cycle
    # -------------------------------
    df["Avg_Gap"] = (
        df.groupby("CustomerPhone")["Days_Since_Last"]
        .transform("mean")
    )

    df["Gap_Std"] = (
        df.groupby("CustomerPhone")["Days_Since_Last"]
        .transform("std")
    )

    # -------------------------------
    # Rolling visit counts
    # -------------------------------
    df["Visits_Last_14"] = 0
    df["Visits_Last_30"] = 0
    df["Visits_Last_60"] = 0

    for phone, group in df.groupby("CustomerPhone"):
        dates = group["Date"].values
        idx = group.index

        for i in range(len(dates)):
            current_date = dates[i]

            df.loc[idx[i], "Visits_Last_14"] = (
                (dates >= current_date - np.timedelta64(14, "D")) &
                (dates <= current_date)
            ).sum()

            df.loc[idx[i], "Visits_Last_30"] = (
                (dates >= current_date - np.timedelta64(30, "D")) &
                (dates <= current_date)
            ).sum()

            df.loc[idx[i], "Visits_Last_60"] = (
                (dates >= current_date - np.timedelta64(60, "D")) &
                (dates <= current_date)
            ).sum()

    # -------------------------------
    # Calendar features
    # -------------------------------
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
    df["Hour"] = df["Time"].dt.hour.fillna(12)

    # -------------------------------
    # Overdue flag
    # -------------------------------
    df["Overdue"] = (
        df["Days_Since_Last"] > df["Avg_Gap"]
    ).astype(int)

    # -------------------------------
    # Target (book within 7 days)
    # -------------------------------
    df["Booked_Next_Week"] = (
        df.groupby("CustomerPhone")["Date"]
        .shift(-1)
        .sub(df["Date"])
        .dt.days
        .le(7)
        .astype(int)
    )

    # Fill missing
    df["Days_Since_Last"] = df["Days_Since_Last"].fillna(999)
    df["Avg_Gap"] = df["Avg_Gap"].fillna(999)
    df["Gap_Std"] = df["Gap_Std"].fillna(0)

    df = df.dropna(subset=["Booked_Next_Week"])

    return df