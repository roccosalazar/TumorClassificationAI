import unittest
import pandas as pd
import numpy as np
from preprocessing import MissingValuesHandler, MissingValuesStrategyManager

class TestMissingValuesHandler(unittest.TestCase):

    def setUp(self):
        # Dataset di esempio
        self.data = pd.DataFrame({
            "A": [1, 2, np.nan, 4],
            "B": [np.nan, 2, 3, 4],
            "classtype_v1": [1, np.nan, 3, 4],
            "C": ["10", "20", "30", "40"]
        })

    def test_convert_numeric_columns(self):
        converted_data = MissingValuesHandler.convert_numeric_columns(self.data)
        self.assertTrue(pd.api.types.is_numeric_dtype(converted_data["C"]))
        expected = pd.Series([10.0, 20.0, 30.0, 40.0], dtype='float64', name="C")
        pd.testing.assert_series_equal(converted_data["C"], expected)

    def test_remove_rows_with_missing_classtype(self):
        filtered_data = MissingValuesHandler.remove_rows_with_missing_classtype(self.data)
        self.assertEqual(len(filtered_data), 3)
        self.assertFalse(filtered_data["classtype_v1"].isnull().any())

    def test_remove_missing_rows(self):
        filtered_data = MissingValuesHandler.remove_rows_with_missing_classtype(self.data)
        cleaned_data = MissingValuesHandler.remove_missing_rows(filtered_data)
        self.assertEqual(len(cleaned_data), 1)
        self.assertFalse(cleaned_data.isnull().any().any())

    def test_fill_missing_with_mean(self):
        numeric_data = MissingValuesHandler.convert_numeric_columns(self.data)
        filtered_data = MissingValuesHandler.remove_rows_with_missing_classtype(numeric_data)
        filled_data = MissingValuesHandler.fill_missing_with_mean(filtered_data)
        self.assertAlmostEqual(filled_data["A"].iloc[1], filtered_data["A"].mean(skipna=True))
        self.assertAlmostEqual(filled_data["B"].iloc[0], filtered_data["B"].mean(skipna=True))

    def test_fill_missing_with_median(self):
        numeric_data = MissingValuesHandler.convert_numeric_columns(self.data)
        filtered_data = MissingValuesHandler.remove_rows_with_missing_classtype(numeric_data)
        filled_data = MissingValuesHandler.fill_missing_with_median(filtered_data)
        self.assertAlmostEqual(filled_data["A"].iloc[1], filtered_data["A"].median(skipna=True))
        self.assertAlmostEqual(filled_data["B"].iloc[0], filtered_data["B"].median(skipna=True))

    def test_fill_missing_with_mode(self):
        numeric_data = MissingValuesHandler.convert_numeric_columns(self.data)
        filtered_data = MissingValuesHandler.remove_rows_with_missing_classtype(numeric_data)
        filled_data = MissingValuesHandler.fill_missing_with_mode(filtered_data)
        self.assertEqual(filled_data["A"].iloc[1], 1)
        self.assertEqual(filled_data["B"].iloc[0], 3)

    def test_strategy_remove(self):
        result = MissingValuesStrategyManager.handle_missing_values("remove", self.data.copy())
        self.assertEqual(len(result), 1)
        self.assertFalse(result.isnull().any().any())

    def test_strategy_mean(self):
        result = MissingValuesStrategyManager.handle_missing_values("mean", self.data.copy())
        self.assertFalse(result.isnull().any().any())
        self.assertAlmostEqual(result["A"].iloc[1], result["A"].mean(skipna=True))

    def test_strategy_median(self):
        result = MissingValuesStrategyManager.handle_missing_values("median", self.data.copy())
        self.assertFalse(result.isnull().any().any())
        self.assertAlmostEqual(result["A"].iloc[1], result["A"].median(skipna=True))

    def test_strategy_mode(self):
        result = MissingValuesStrategyManager.handle_missing_values("mode", self.data.copy())
        self.assertFalse(result.isnull().any().any())
        self.assertEqual(result["A"].iloc[1], 1)

    def test_invalid_strategy(self):
        with self.assertRaises(ValueError):
            MissingValuesStrategyManager.handle_missing_values("invalid", self.data)

if __name__ == "__main__":
    unittest.main()
