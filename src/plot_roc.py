"""Plot empirical ROC curves for appointment-return windows."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from src.features import FEATURES, build_features
from src.train import new_pipeline


def plot_roc(data_path, output_dir="reports", observation_end=None):
    windows = (7, 14, 21, 31, 38, 45, 60)
    data = pd.read_csv(data_path, dtype={"Phone": "string", "CustomerPhone": "string"})
    base = build_features(data, observation_end=observation_end)
    end = pd.Timestamp(observation_end).normalize() if observation_end else base["Date"].max()
    eligible = base[base["Date"] + pd.Timedelta(days=max(windows)) <= end]
    if eligible.empty:
        raise ValueError("Need historical appointments with 60 days of observed follow-up.")
    cutoff = eligible["Date"].quantile(0.7)
    train = eligible[eligible["Date"] + pd.Timedelta(days=max(windows)) <= cutoff]
    test = eligible[eligible["Date"] > cutoff]
    if train.empty or test.empty:
        raise ValueError("Not enough history for the temporal split and 60-day purge.")
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
    results = []
    curves = []
    plotted = 0
    for days in windows:
        y_train = train["Days_To_Next_Appointment"].le(days).astype(int)
        y_test = test["Days_To_Next_Appointment"].le(days).astype(int)
        row = {"Window_Days": days, "Train_Rows": len(train), "Test_Rows": len(test), "ROC_AUC": None, "Status": "ok"}
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            row["Status"] = "One target class; ROC unavailable"
            results.append(row)
            continue
        model = new_pipeline()
        model.fit(train[FEATURES], y_train)
        scores = model.predict_proba(test[FEATURES])[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, scores)
        auc = roc_auc_score(y_test, scores)
        row["ROC_AUC"] = auc
        results.append(row)
        curves.append(pd.DataFrame({"Window_Days": days, "False_Positive_Rate": fpr, "True_Positive_Rate": tpr, "Threshold": thresholds}))
        ax.step(fpr, tpr, where="post", linewidth=2, label=f"{days} days — AUC {auc:.4f}")
        plotted += 1
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random ranking — AUC 0.5000")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate (non-returners flagged)", ylabel="True positive rate (returners identified)", title="Barbershop ML: ROC curves on later appointments")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=9)
    fig.suptitle(f"Train {len(train):,} | Test {len(test):,} | Follow-up and purge: 60 days", fontsize=10)
    if not plotted:
        ax.text(0.1, 0.8, "No windows with both classes; see roc_auc.csv")
    fig.savefig(folder / "roc_curves.png", dpi=180)
    fig.savefig(folder / "roc_curves.pdf")
    plt.close(fig)
    pd.DataFrame(results).to_csv(folder / "roc_auc.csv", index=False)
    if curves:
        pd.concat(curves, ignore_index=True).to_csv(folder / "roc_points.csv", index=False)
    print(pd.DataFrame(results).to_string(index=False))
    print(f"Saved ROC chart to {folder / 'roc_curves.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/appointments.csv")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--observation-end")
    args = parser.parse_args()
    plot_roc(args.data, args.output_dir, args.observation_end)
