# Barbershop ML

A Python baseline for exploring repeat appointment patterns and exporting customer segments for a barbershop. It extracts appointment requests from local HTML exports, builds visit-history features, and trains logistic regression to estimate whether a customer has another recorded appointment within seven days of a visit.

This is an exploratory project. Appointment requests are not proof of completed visits, and scores have not been validated for production use. No accuracy or business-impact claims are made.

## Setup

Use Python 3.10 or newer. Run commands from the repository root.

```bash
git clone https://github.com/Ed-project-1/barbershop-ml.git
cd barbershop-ml
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

In VS Code, open this folder and select `.venv/bin/python` with **Python: Select Interpreter**.

## Data and extraction

Place your own HTML exports in `html_exports/`, then run:

```bash
python extract_from_html.py
```

The extractor looks for message `div` elements with class `text` containing `You have a new appointment request!`, followed by Customer, Date, Time, and Phone fields. It writes `data/appointments.csv`, removes duplicate appointments, and excludes records without a phone. Missing phones are filled by customer name only when that name has one known phone; names are not reliable unique identifiers. Phone normalization retains the last nine digits and is intended for the original Polish data, not international numbers.

Alternatively, provide a CSV with these columns:

| Column | Meaning | Example |
| --- | --- | --- |
| Customer | Customer name | Demo Customer |
| Date | Appointment date; ISO format recommended | 2025-01-05 |
| Time | 12-hour appointment time | 10:00 AM |
| Phone | Phone text, when using the extractor | 000000000 |
| CustomerPhone | Stable customer identifier used for grouping | Demo Customer \| 000000000 |

Raw exports, datasets, campaign files, local environments, and generated models are ignored by Git. Real customer data is not included.

## Compare appointment-return windows

Each historical appointment is an anchor. The target is whether the same customer has their next recorded appointment within 7, 14, 21, or 31 days after that anchor. These are cumulative windows (within 14 days includes returns within 7 days), not exclusive intervals. They describe recorded appointments, not confirmed attendance or the date when a booking was made.

```bash
python -m src.train
# If your complete export covers activity through a known end date:
python -m src.train --observation-end 2026-10-13
```

The command prints a comparison table and saves `reports/window_auc.csv` and `reports/window_auc.md`. Columns include the window, training/test rows, test returners, return rate, ROC AUC, and status. Missing AUC means a split lacks enough rows or one target class.

Every window uses the same appointments and temporal split. All anchors must have 31 days of follow-up. The earlier 70% date boundary defines the split, and a common 31-day purge prevents training outcomes crossing into the test period. Without an explicit observation end, the latest recorded appointment date is used; the export must cover that whole period continuously. AUC values will therefore differ from the old seven-day-only evaluation.

Gap statistics and visit counts use history available at the anchor. Future appointment information is used only for the target. Invalid identities/dates and duplicate appointments are excluded. No customer dataset is bundled; real AUC values require your local CSV.

Review the table before choosing a window appropriate to the salon. AUC across different targets does not by itself establish the best business horizon. Choosing the largest test AUC is exploratory and requires a fresh holdout before claiming performance.

## Fit the chosen window, then predict

For example, after reviewing the table, explicitly select 21 days:

```bash
python -m src.train --horizon 21
python -m src.predict
```

The chosen model is fitted on all fully observed rows for that horizon and saved with its horizon metadata. Retrain older model files before prediction. Logistic regression uses StandardScaler and no class reweighting; output probabilities still need calibration checks before operational use.

## Campaign exports

After training:

```bash
python -m src.predict
```

Outputs are written to `campaign_logs/`:

- `active_campaign.csv`: latest recorded visit score at least 0.6 and visit within 120 days.
- `reactivation_campaign.csv`: latest visit older than 120 days and at most 420 days.

These files are local planning outputs; this command sends no messages. Scores are evaluated at each customer's latest recorded visit, not at today's date. The exports include Prediction_Window_Days and Window_End_Date; a window that has already passed is a historical score, not an upcoming prediction. Treat the segments as exploratory; a daily outreach model needs daily customer snapshots, an appropriate target, and fresh validation.

## Checks

```bash
python -m unittest discover -s tests -v
python -m compileall -q src extract_from_html.py
```

Tests cover historical feature stability, incomplete labels, invalid records, and same-day ordering.

## Optional Telegram connectivity test

```bash
python -m pip install -r requirements-telegram.txt
cp .env.example .env
# Set your own token and chat ID locally.
python -m src.test_bot
```

This explicitly sends one test message to the configured chat. It is separate from model training and prediction. Never commit credentials. The former tracked `.env` and source ZIP are removed from this branch, but remain in earlier Git history; rotate any real token they contained.

## Code layout

| Path | Purpose |
| --- | --- |
| `extract_from_html.py` | HTML appointment extraction |
| `src/features.py` | Shared feature definitions and target construction |
| `src/train.py` | Temporal evaluation and model saving |
| `src/predict.py` | Local customer segment exports |
| `src/test_bot.py` | Optional explicit Telegram test |
| `tests/test_features.py` | Feature regression tests |

Built with pandas, NumPy, scikit-learn, joblib, and Beautiful Soup.

## Predict the next appointment with survival analysis

Install optional survival dependencies into your active environment:

```bash
python -m pip install -r requirements-survival.txt
```

Use a complete export and its actual last covered date, not simply today's date. For example, if the export covers all appointment activity through 13 October 2026:

```bash
python -m src.survival train --observation-end 2026-10-13
python -m src.survival predict --observation-end 2026-10-13
```

Outputs in `reports/survival/`:

- `survival_metrics.csv`: validation results for Cox and random survival forest, final test results for the selected model, and a population Kaplan–Meier baseline.
- `survival_model.pkl`: selected model refitted on all available history for deployment.
- `customer_predictions.csv`: one row per customer, last appointment, elapsed days, return probabilities in the next 7/14/21/31/38/45/60 days, and conditional P25/median/P75 return dates.
- `model_failures.csv`: fitting or validation failures, when present.

The model treats observed next appointments as events and unfinished intervals as right-censored. At each temporal training boundary, future returns are censored at that boundary; they are not visible to training. Historical covariates are measured at the appointment anchor. Zero-day intervals are excluded because this model measures whole-day return times.

Models are selected by the lowest mean validation Brier score across the seven horizons, then evaluated on a separate later test period. Cox has fixed ridge regularization; the forest uses fixed settings. The 60% and 80% date boundaries define validation and test periods. Evaluation anchors require a complete 60-day follow-up inside their period. This may require several years of history or different predefined split dates for short datasets; the script stops when there is insufficient history rather than using an invalid split.

Reported AUC is horizon-specific AUC on fully observed outcomes. Brier scores measure probability error (lower is better); because all evaluated horizons have complete follow-up, ordinary binary AUC/Brier apply without censoring weights. They differ from the earlier 70/30 logistic-regression table. The deployment refit has no new test score; use the saved held-out results to judge performance.

Prediction date must equal the complete-export observation end. Refresh the export to predict from today; a stale export cannot establish that someone has not returned since it ended. Historical scoring with a model trained after the requested date is rejected.

Conditional return probability is `1 - S(elapsed + horizon) / S(elapsed)`. Thus someone last seen 40 days ago is scored for the coming week conditional on having no recorded return during those 40 days. P25/median/P75 dates are model-derived return-time quantiles, not guaranteed appointments or confidence intervals. Blank quantiles mean the cumulative probability never reached that quantile within model support. Blank probabilities mean the horizon is beyond learned follow-up support or conditioning is numerically unsupported; blanks are not zero likelihood.

Appointment requests remain proxies for appointments, not attendance. Identity changes, name/phone collisions, missing exports and future scheduled appointments can distort intervals. Supply only historical appointments through the observation end; the data validator rejects later dates. Cox proportional-hazards assumptions and probability calibration need checking on real data before operational use. The forest comparison helps assess nonlinear patterns but does not establish those assumptions.

```bash
python -m unittest discover -s tests -v
```
