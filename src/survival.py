"""Compare Cox and survival forest, then predict conditional return times."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.nonparametric import kaplan_meier_estimator
from src.features import build_features

HORIZONS = (7, 14, 21, 31, 38, 45, 60)
COVARIATES = ["Days_Since_Last", "Total_Visits", "Avg_Gap", "Gap_Std",
              "Visits_Last_14", "Visits_Last_30", "Visits_Last_60", "DayOfWeek", "Month", "Hour"]


def intervals(features, end):
    """Administratively censor every interval at the specified observation boundary."""
    df = features[features["Date"] <= end].copy()
    next_date = df["Next_Appointment_Date"]
    df["Event"] = next_date.notna() & next_date.le(end)
    stop = next_date.where(df["Event"], end)
    df["Duration"] = (stop - df["Date"]).dt.days
    return df[df["Duration"] > 0].copy()


def fit_model(name, rows):
    if len(rows) < 20 or rows["Event"].sum() < 5:
        raise ValueError("Need at least 20 positive-duration intervals and five observed returns.")
    estimator = CoxPHSurvivalAnalysis(alpha=0.1) if name == "cox" else RandomSurvivalForest(
        n_estimators=150, min_samples_leaf=10, max_features="sqrt", random_state=42, n_jobs=-1)
    model = make_pipeline(VarianceThreshold(), StandardScaler(), estimator)
    model.fit(rows[COVARIATES], Surv.from_arrays(rows["Event"].astype(bool), rows["Duration"].astype(float)))
    return model


def at_time(curve, day):
    if day == 0:
        return 1.0
    if day > curve.x[-1]:
        return np.nan
    if day < curve.x[0]:
        return 1.0
    return float(curve(day))


def conditional_probabilities(curve, age, horizons=HORIZONS):
    denominator = at_time(curve, age)
    if not np.isfinite(denominator) or denominator <= 1e-10:
        return [np.nan] * len(horizons)
    return [float(np.clip(1 - at_time(curve, age + day) / denominator, 0, 1))
            if np.isfinite(at_time(curve, age + day)) else np.nan for day in horizons]


def evaluate(model, test, name, phase):
    curves = model.predict_survival_function(test[COVARIATES])
    rows = []
    for day in HORIZONS:
        truth = test["Days_To_Next_Appointment"].le(day).astype(int).to_numpy()
        probability = np.array([1 - at_time(curve, day) for curve in curves])
        if not np.isfinite(probability).all():
            rows.append({"Phase": phase, "Model": name, "Days": day, "Rows": len(test), "AUC": None, "Brier": None, "Status": "Horizon beyond model support"})
            continue
        rows.append({"Phase": phase, "Model": name, "Days": day, "Rows": len(test),
                     "AUC": roc_auc_score(truth, probability) if np.unique(truth).size == 2 else None,
                     "Brier": brier_score_loss(truth, probability), "Status": "ok" if np.unique(truth).size == 2 else "AUC undefined: one outcome class"})
    return rows


def baseline_metrics(training, test, phase):
    times, survival = kaplan_meier_estimator(training["Event"].to_numpy(dtype=bool), training["Duration"].to_numpy(dtype=float))
    result = []
    for day in HORIZONS:
        index = np.searchsorted(times, day, side="right") - 1
        probability = 1 - survival[index] if index >= 0 else 0.0
        truth = test["Days_To_Next_Appointment"].le(day).astype(int).to_numpy()
        result.append({"Phase": phase, "Model": "kaplan_meier_baseline", "Days": day, "Rows": len(test),
                       "AUC": .5 if np.unique(truth).size == 2 else None,
                       "Brier": brier_score_loss(truth, np.full(len(truth), probability)), "Status": "ok"})
    return result


def train(data, end, output):
    features = build_features(data, observation_end=end)
    if features.empty:
        raise ValueError("No valid appointments.")
    boundaries = features["Date"].quantile([0.6, 0.8])
    validation_start, test_start = boundaries.iloc[0], boundaries.iloc[1]
    validation = features[(features["Date"] > validation_start) & (features["Date"] + pd.Timedelta(days=60) <= test_start)]
    test = features[(features["Date"] > test_start) & (features["Date"] + pd.Timedelta(days=60) <= end)]
    if len(validation) < 20 or len(test) < 20:
        raise ValueError("Need more history: each validation/test period needs 20 anchors and complete 60-day follow-up.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    all_rows = baseline_metrics(intervals(features, validation_start), validation, "validation")
    choices = {}
    failures = []
    for name in ("cox", "forest"):
        try:
            model = fit_model(name, intervals(features, validation_start))
            metrics = evaluate(model, validation, name, "validation")
            values = [r["Brier"] for r in metrics if r["Brier"] is not None]
            if len(values) != len(HORIZONS):
                raise ValueError("All seven horizons must be supported for model comparison.")
            choices[name] = float(np.mean(values))
            all_rows.extend(metrics)
        except (ValueError, ArithmeticError) as error:
            failures.append({"Model": name, "Error": str(error)})
    pd.DataFrame(failures, columns=["Model", "Error"]).to_csv(output / "model_failures.csv", index=False)
    if not choices:
        raise ValueError(f"No model passed validation: {failures}")
    selected = min(choices, key=choices.get)
    # Selection happens before opening the final test period.
    evaluation_model = fit_model(selected, intervals(features, test_start))
    all_rows.extend(evaluate(evaluation_model, test, selected, "test"))
    all_rows.extend(baseline_metrics(intervals(features, test_start), test, "test"))
    table = pd.DataFrame(all_rows)
    table.to_csv(output / "survival_metrics.csv", index=False)
    deployment_model = fit_model(selected, intervals(features, end))
    artifact = {"model": deployment_model, "name": selected, "features": COVARIATES,
                "observation_end": str(end.date()), "validation_start": str(validation_start), "test_start": str(test_start)}
    joblib.dump(artifact, output / "survival_model.pkl")
    print(table.to_string(index=False))
    print(f"Selected {selected} by mean validation Brier score: {choices[selected]:.4f}")
    return artifact


def predict(artifact, data, as_of, output):
    # A prediction from today needs a complete history through today.
    features = build_features(data, observation_end=as_of)
    latest = features.groupby("CustomerPhone", sort=False).tail(1).copy().reset_index(drop=True)
    if latest.empty:
        raise ValueError("No customers to score.")
    curves = artifact["model"].predict_survival_function(latest[artifact["features"]])
    rows = []
    for (_, row), curve in zip(latest.iterrows(), curves):
        age = (as_of - row["Date"]).days
        probabilities = conditional_probabilities(curve, age)
        result = {"CustomerID": row.get("CustomerID", row["CustomerPhone"]), "Customer": row["Customer"], "CustomerPhone": row["CustomerPhone"],
                  "Last_Appointment": str(row["Date"].date()), "As_Of": str(as_of.date()),
                  "Days_Since_Last": age, "Model": artifact["name"]}
        for day, probability in zip(HORIZONS, probabilities):
            result[f"Return_Probability_Next_{day}_Days"] = probability
        denominator = at_time(curve, age)
        supported = np.isfinite(denominator) and denominator > 1e-10
        for quantile, label in ((0.25, "P25"), (0.5, "Median"), (0.75, "P75")):
            date = None
            if supported:
                grid = np.arange(age + 1, int(curve.x[-1]) + 1)
                found = [int(day) for day in grid if 1 - at_time(curve, day) / denominator >= quantile]
                if found:
                    date = str((row["Date"] + pd.Timedelta(days=found[0])).date())
            result[f"Estimated_Next_Date_{label}"] = date
        result["Status"] = "ok" if all(np.isfinite(probabilities)) else "Some horizons unsupported; blank estimates are not zero probability"
        rows.append(result)
    predictions = pd.DataFrame(rows)
    predictions.to_csv(output, index=False)
    print(f"Saved predictions for {len(predictions)} customers to {output}")
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "predict"))
    parser.add_argument("--data", default="data/appointments.csv")
    parser.add_argument("--output-dir", default="reports/survival")
    parser.add_argument("--observation-end", required=True, help="Last date covered by a complete export, YYYY-MM-DD")
    parser.add_argument("--as-of", help="Prediction date; defaults to observation end")
    args = parser.parse_args()
    end = pd.Timestamp(args.observation_end).normalize()
    data = pd.read_csv(args.data, dtype={"Phone": "string", "CustomerPhone": "string"})
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.action == "train":
        train(data, end, output)
    else:
        as_of = pd.Timestamp(args.as_of).normalize() if args.as_of else end
        if as_of != end:
            raise ValueError("Prediction date must equal complete-export observation end; refresh the data to predict from today.")
        artifact = joblib.load(output / "survival_model.pkl")
        if pd.Timestamp(artifact["observation_end"]) > as_of:
            raise ValueError("Model was trained after the requested prediction date.")
        predict(artifact, data, as_of, output / "customer_predictions.csv")


if __name__ == "__main__":
    main()
