"""Compare return windows measured from each recorded appointment."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.features import FEATURES, build_features

WINDOWS = (7, 14, 21, 31)


def new_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=42)),
    ])


def compare_windows(data_path, output_dir="reports", observation_end=None):
    appointments = pd.read_csv(data_path, dtype={"Phone": "string", "CustomerPhone": "string"})
    base = build_features(appointments, observation_end=observation_end)
    end = pd.Timestamp(observation_end).normalize() if observation_end else base["Date"].max()
    # Identical rows and temporal boundaries for every window.
    eligible = base[base["Date"] + pd.Timedelta(days=max(WINDOWS)) <= end]
    if eligible.empty:
        raise ValueError("Need appointment history with at least 31 days of follow-up.")
    cutoff = eligible["Date"].quantile(0.7)
    train_mask = eligible["Date"] + pd.Timedelta(days=max(WINDOWS)) <= cutoff
    test_mask = eligible["Date"] > cutoff
    train = eligible[train_mask]
    test = eligible[test_mask]
    results = []
    for days in WINDOWS:
        labels = eligible["Days_To_Next_Appointment"].le(days).astype(int)
        y_train, y_test = labels.loc[train.index], labels.loc[test.index]
        row = {"Window_Days": days, "Train_Rows": len(train), "Test_Rows": len(test),
               "Test_Returners": int(y_test.sum()),
               "Test_Return_Rate": float(y_test.mean()) if len(test) else None,
               "ROC_AUC": None, "Status": "ok"}
        if train.empty or test.empty:
            row["Status"] = "Insufficient rows in temporal split"
        elif y_train.nunique() < 2:
            row["Status"] = "Training target has only one class"
        else:
            pipeline = new_pipeline()
            pipeline.fit(train[FEATURES], y_train)
            scores = pipeline.predict_proba(test[FEATURES])[:, 1]
            if y_test.nunique() < 2:
                row["Status"] = "Test target has only one class; AUC undefined"
            else:
                row["ROC_AUC"] = float(roc_auc_score(y_test, scores))
        results.append(row)
    table = pd.DataFrame(results)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(folder / "window_auc.csv", index=False)
    metadata = f"Observation end: {end.date()} | Split boundary: {cutoff} | Common follow-up and purge: 31 days"
    lines = [metadata, "", "| Days after appointment | Train rows | Test rows | Test returners | Return rate | ROC AUC | Status |",
             "|---|---|---|---|---|---|---|"]
    for row in results:
        rate = f"{row['Test_Return_Rate']:.1%}" if row['Test_Return_Rate'] is not None else "N/A"
        auc = f"{row['ROC_AUC']:.4f}" if row['ROC_AUC'] is not None else "N/A"
        lines.append(f"| {row['Window_Days']} | {row['Train_Rows']} | {row['Test_Rows']} | {row['Test_Returners']} | {rate} | {auc} | {row['Status']} |")
    report = "\n".join(lines) + "\n"
    (folder / "window_auc.md").write_text(report)
    print(report)
    return table


def train_model(data_path, model_path, horizon_days=31, observation_end=None):
    """Fit the explicitly selected window after reviewing the comparison."""
    df = build_features(pd.read_csv(data_path, dtype={"Phone": "string", "CustomerPhone": "string"}), horizon_days, observation_end)
    end = pd.Timestamp(observation_end).normalize() if observation_end else df["Date"].max()
    df = df[df["Date"] + pd.Timedelta(days=horizon_days) <= end]
    if df.empty or df["Returned_Within_Window"].nunique() < 2:
        raise ValueError("Need fully observed history with both target classes.")
    pipeline = new_pipeline()
    pipeline.fit(df[FEATURES], df["Returned_Within_Window"].astype(int))
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "horizon_days": horizon_days, "features": FEATURES}, model_path)
    print(f"Saved {horizon_days}-day model fitted on {len(df)} rows to {model_path}")
    return pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/appointments.csv")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--observation-end", help="Last date covered by the complete export, YYYY-MM-DD")
    parser.add_argument("--horizon", type=int, choices=WINDOWS, help="Optionally fit this chosen window after comparison")
    parser.add_argument("--model", default="models/logistic_model.pkl")
    args = parser.parse_args()
    compare_windows(args.data, args.output_dir, args.observation_end)
    if args.horizon:
        train_model(args.data, args.model, args.horizon, args.observation_end)


if __name__ == "__main__":
    main()
