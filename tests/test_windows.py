import tempfile
import unittest
from pathlib import Path
import pandas as pd
from src.features import build_features
from src.train import compare_windows


class WindowTests(unittest.TestCase):
    def test_boundary_labels(self):
        data = pd.DataFrame({"Customer": ["Demo"] * 4, "CustomerPhone": ["Demo"] * 4,
                             "Date": ["2025-01-01", "2025-01-15", "2025-02-05", "2025-03-08"], "Time": ["10:00 AM"] * 4})
        for window, expected in [(7, [0, 0, 0]), (14, [1, 0, 0]), (21, [1, 1, 0]), (31, [1, 1, 1])]:
            result = build_features(data, window)
            self.assertEqual(result["Returned_Within_Window"].iloc[:3].tolist(), expected)
            self.assertTrue(pd.isna(result["Returned_Within_Window"].iloc[-1]))

    def test_comparison_uses_same_rows_and_writes_table(self):
        rows = []
        for customer, gap in enumerate([5, 10, 18, 25, 40, 50]):
            for date in pd.date_range("2024-01-01", "2025-12-01", freq=f"{gap}D"):
                rows.append({"Customer": f"Demo {customer}", "CustomerPhone": f"ID {customer}", "Date": date, "Time": "10:00 AM"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(rows).to_csv(root / "data.csv", index=False)
            table = compare_windows(root / "data.csv", root / "reports", "2025-12-31")
            self.assertEqual(table["Window_Days"].tolist(), [7, 14, 21, 31])
            self.assertEqual(table["Train_Rows"].nunique(), 1)
            self.assertEqual(table["Test_Rows"].nunique(), 1)
            self.assertTrue(table["ROC_AUC"].between(0, 1).all())
            self.assertTrue((root / "reports/window_auc.md").exists())


if __name__ == "__main__":
    unittest.main()
