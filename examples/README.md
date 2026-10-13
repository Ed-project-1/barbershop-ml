# Synthetic runnable example

`appointments_sample.csv` contains fictional appointments for 50 customers. Every phone starts with zero and is a mock identifier, not a number to contact. The fixture uses seed 42, with history from 2020 and complete simulated observation through **2026-05-28**. It includes different visit cycles, one-time customers, name aliases, one-digit phone typos and one-day reschedules. No business customer records are included.

Run these commands from the repository root in your activated Python environment:

```bash
python -m pip install -r requirements-survival.txt
python -m src.clean_customers --input examples/appointments_sample.csv --output data/example_appointments_cleaned.csv --audit-dir reports/example_customer_cleaning
python -m src.survival train --data data/example_appointments_cleaned.csv --observation-end 2026-05-28 --output-dir reports/example_survival
python -m src.survival predict --data data/example_appointments_cleaned.csv --observation-end 2026-05-28 --output-dir reports/example_survival
```

Inspect:

- `reports/example_customer_cleaning/identity_map.csv` for recovered aliases.
- `reports/example_customer_cleaning/merge_rules.csv` for merge reasons.
- `reports/example_survival/survival_metrics.csv` for demo evaluation.
- `reports/example_survival/customer_predictions.csv` for demo predictions.

The separate output directories prevent replacing models trained on your real data. Prediction is from the simulated observation date, not from today. Metrics measure synthetic patterns and are not evidence of business performance. Read phones as text to preserve their leading zero.

Regenerate the fixture with `python scripts/generate_example.py`.
