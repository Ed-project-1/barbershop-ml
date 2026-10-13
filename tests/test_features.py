import unittest
import pandas as pd
from src.features import FEATURES, build_features


def appointments(dates):
    return pd.DataFrame({"Customer": ["Demo"] * len(dates), "CustomerPhone": ["Demo | 000000000"] * len(dates), "Date": dates, "Time": ["10:00 AM"] * len(dates)})


class FeatureTests(unittest.TestCase):
    def test_future_visits_do_not_change_past_features(self):
        past = appointments(["2025-01-01", "2025-01-05", "2025-01-15"])
        original = past.copy(deep=True)
        before = build_features(past)
        after = build_features(appointments(["2025-01-01", "2025-01-05", "2025-01-15", "2025-02-01"]))
        pd.testing.assert_frame_equal(before[FEATURES], after.iloc[:3][FEATURES])
        pd.testing.assert_frame_equal(past, original)

    def test_unobserved_last_target_is_missing(self):
        result = build_features(appointments(["2025-01-01", "2025-01-05", "2025-01-15"]))
        self.assertEqual(result.loc[0, "Booked_Next_Week"], 1)
        self.assertEqual(result.loc[1, "Booked_Next_Week"], 0)
        self.assertTrue(pd.isna(result.loc[2, "Booked_Next_Week"]))

    def test_missing_identity_and_invalid_dates_are_removed(self):
        data = appointments(["2025-01-01", "bad", "2025-01-03"])
        data.loc[2, "CustomerPhone"] = None
        self.assertEqual(len(build_features(data)), 1)

    def test_same_day_later_visit_not_counted_in_first_row(self):
        data = appointments(["2025-01-01", "2025-01-01"])
        data.loc[1, "Time"] = "02:00 PM"
        self.assertEqual(build_features(data)["Visits_Last_14"].tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()
