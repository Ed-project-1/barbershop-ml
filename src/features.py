"""Causal visit-history features and configurable return-window targets."""
import numpy as np
import pandas as pd

FEATURES = [
    "Days_Since_Last", "Total_Visits", "Avg_Gap", "Gap_Std",
    "Visits_Last_14", "Visits_Last_30", "Visits_Last_60", "Overdue",
    "DayOfWeek", "Month", "IsWeekend", "Hour",
]


def build_features(appointments, horizon_days=7, observation_end=None):
    """Use only visits at or before each row; retain unobserved targets as missing."""
    if horizon_days < 1:
        raise ValueError("horizon_days must be positive.")
    required = {"Customer", "CustomerPhone", "Date", "Time"}
    missing = required - set(appointments.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    df = appointments.copy()
    df["Customer"] = df["Customer"].astype("string").str.strip()
    df["CustomerPhone"] = df["CustomerPhone"].astype("string").str.strip()
    if "CustomerID" in df:
        ids = df["CustomerID"].astype("string").str.strip()
        if ids.isna().any() or ids.eq("").any():
            raise ValueError("Cleaned data contains missing CustomerIDs.")
        df["CustomerPhone"] = ids
    df = df[df["CustomerPhone"].notna() & df["CustomerPhone"].ne("")].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    hours = pd.to_datetime(df["Time"], format="%I:%M %p", errors="coerce")
    df["Hour"] = hours.dt.hour.fillna(12)
    df["_minutes"] = (hours.dt.hour * 60 + hours.dt.minute).fillna(720)
    df = df.dropna(subset=["Date"]).sort_values(
        ["CustomerPhone", "Date", "_minutes"], kind="stable"
    ).drop_duplicates(["CustomerPhone", "Date", "_minutes"]).reset_index(drop=True)
    groups = df.groupby("CustomerPhone", sort=False)
    df["Days_Since_Last"] = groups["Date"].diff().dt.days
    df["Total_Visits"] = groups.cumcount() + 1
    df["Avg_Gap"] = df.groupby("CustomerPhone")["Days_Since_Last"].transform(
        lambda values: values.expanding().mean()
    )
    df["Gap_Std"] = df.groupby("CustomerPhone")["Days_Since_Last"].transform(
        lambda values: values.expanding().std()
    )
    for window in (14, 30, 60):
        df[f"Visits_Last_{window}"] = 0
    for _, group in df.groupby("CustomerPhone", sort=False):
        dates = group["Date"].to_numpy()
        for window in (14, 30, 60):
            starts = np.searchsorted(dates, dates - np.timedelta64(window, "D"))
            df.loc[group.index, f"Visits_Last_{window}"] = np.arange(len(group)) + 1 - starts
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
    df["Overdue"] = df["Days_Since_Last"].gt(df["Avg_Gap"]).astype(int)
    next_date = df.groupby("CustomerPhone")["Date"].shift(-1)
    end = pd.Timestamp(observation_end).normalize() if observation_end is not None else df["Date"].max()
    if not df.empty and end < df["Date"].max():
        raise ValueError("observation_end cannot precede the latest appointment date.")
    df["Next_Appointment_Date"] = next_date
    df["Days_To_Next_Appointment"] = (next_date - df["Date"]).dt.days
    target = df["Days_To_Next_Appointment"].le(horizon_days).astype("Int64")
    observed = (next_date.notna() & next_date.le(df["Date"] + pd.Timedelta(days=horizon_days))) | (df["Date"] + pd.Timedelta(days=horizon_days) <= end)
    df["Returned_Within_Window"] = target.where(observed)
    if horizon_days == 7:
        df["Booked_Next_Week"] = df["Returned_Within_Window"]
    df["Days_Since_Last"] = df["Days_Since_Last"].fillna(999)
    df["Avg_Gap"] = df["Avg_Gap"].fillna(999)
    df["Gap_Std"] = df["Gap_Std"].fillna(0)
    return df.drop(columns="_minutes")
