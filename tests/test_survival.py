import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sksurv.functions import StepFunction
from src.features import build_features
from src.survival import intervals, conditional_probabilities, train, predict


class SurvivalTests(unittest.TestCase):
    def test_training_censors_returns_after_boundary(self):
        data = pd.DataFrame({"Customer": ["Demo"] * 2, "CustomerPhone": ["ID"] * 2,
                             "Date": ["2025-01-01", "2025-01-20"], "Time": ["10:00 AM"] * 2})
        features = build_features(data)
        result = intervals(features, pd.Timestamp("2025-01-10"))
        self.assertFalse(result.iloc[0]["Event"])
        self.assertEqual(result.iloc[0]["Duration"], 9)

    def test_conditional_probability_accounts_for_elapsed_time(self):
        curve = StepFunction(np.array([1., 10., 20., 30.]), np.array([1., .8, .4, .2]))
        probabilities = conditional_probabilities(curve, 10, (10, 20, 21))
        self.assertAlmostEqual(probabilities[0], .5)
        self.assertAlmostEqual(probabilities[1], .75)
        self.assertTrue(np.isnan(probabilities[2]))

    def test_end_to_end(self):
        rng = np.random.default_rng(42)
        rows = []
        for customer in range(35):
            date = pd.Timestamp("2021-01-01") + pd.Timedelta(days=int(rng.integers(0, 30)))
            while date < pd.Timestamp("2025-10-01"):
                rows.append({"Customer": f"Demo {customer}", "CustomerPhone": f"ID {customer}", "Date": date, "Time": "10:00 AM"})
                date += pd.Timedelta(days=int(rng.integers(10, 90)))
        data = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = train(data, pd.Timestamp("2025-12-31"), root)
            result = predict(artifact, data, pd.Timestamp("2025-12-31"), root / "predictions.csv")
            self.assertEqual(len(result), 35)
            probabilities = result[[f"Return_Probability_Next_{d}_Days" for d in (7, 14, 21, 31, 38, 45, 60)]]
            self.assertTrue(((probabilities >= 0) & (probabilities <= 1) | probabilities.isna()).all().all())
            for values in probabilities.to_numpy():
                known = values[np.isfinite(values)]
                self.assertTrue((np.diff(known) >= -1e-10).all())
            self.assertTrue((root / "survival_metrics.csv").exists())


if __name__ == "__main__":
    unittest.main()
