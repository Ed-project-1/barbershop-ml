import unittest
import pandas as pd
from src.clean_customers import clean_customers
from src.features import build_features


class CleaningTests(unittest.TestCase):
    def data(self, names, phones):
        return pd.DataFrame({"Customer": names, "Phone": phones, "Date": pd.date_range("2024-01-01", periods=len(names), freq="10D").astype(str), "Time": ["10:00 AM"] * len(names)})

    def test_user_aliases_and_typo(self):
        data = self.data(["Boniface Mwanza", "Boniface", "Boniface", "Bless", "Blessing Adesina", "Blessing", "Blessing"],
                         ["513878640", "513878640", "513878649", "578906056", "578906056", "578906056", "574176465"])
        result, mapping, edges, rejected = clean_customers(data)
        self.assertEqual(result.iloc[:3]["CustomerID"].nunique(), 1)
        self.assertEqual(result.iloc[3:6]["CustomerID"].nunique(), 1)
        self.assertNotEqual(result.iloc[5]["CustomerID"], result.iloc[6]["CustomerID"])
        self.assertEqual(result.iloc[0]["Customer"], "Boniface Mwanza")
        self.assertEqual(result.iloc[3]["Customer"], "Blessing Adesina")
        features = build_features(result)
        self.assertEqual(features.groupby("CustomerID")["Total_Visits"].max().max(), 3)

    def test_full_name_different_numbers_and_unrelated_name(self):
        result, _, _, _ = clean_customers(self.data(["John Doe", "john DOE", "Alice"], ["111111111", "999999999", "111111112"]))
        self.assertEqual(result.iloc[:2]["CustomerID"].nunique(), 1)
        self.assertNotEqual(result.iloc[0]["CustomerID"], result.iloc[2]["CustomerID"])

    def test_duplicates_and_idempotence(self):
        data = self.data(["Boniface Mwanza", "Boniface"], ["+48 513 878 640", "513878640"])
        data.loc[1, "Date"] = data.loc[0, "Date"]
        result, _, _, _ = clean_customers(data)
        self.assertEqual(len(result), 1)
        second, _, _, _ = clean_customers(result)
        self.assertEqual(second["CustomerID"].tolist(), result["CustomerID"].tolist())

    def test_invalid_phones_not_merged(self):
        result, _, _, rejected = clean_customers(self.data(["Demo", "Demo"], ["", "123"]))
        self.assertTrue(result.empty)
        self.assertEqual(len(rejected), 2)


if __name__ == "__main__":
    unittest.main()
