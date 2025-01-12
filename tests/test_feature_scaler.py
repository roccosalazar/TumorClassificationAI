import unittest
import pandas as pd
import numpy as np
from preprocessing import FeatureScaler, FeatureScalerStrategyManager  # Sostituisci con il nome del modulo corretto

class TestFeatureScaler(unittest.TestCase):

    def setUp(self):
        # Dataset di esempio
        self.data = pd.DataFrame({
            "A": [10, 20, 30, 40],
            "B": [1.0, 2.0, 3.0, 4.0],
            "C": [100, 200, 300, 400]
        })
        self.exclude_columns = ["C"]  # Colonne da escludere dallo scaling

    def test_normalize(self):
        # Esegui normalizzazione
        scaled_data = FeatureScaler.normalize(self.data, self.exclude_columns)

        # Controlla che le colonne escluse rimangano invariate
        pd.testing.assert_series_equal(scaled_data["C"], self.data["C"])

        # Controlla che le colonne scalate siano tra 0 e 1
        for col in ["A", "B"]:
            self.assertTrue((scaled_data[col] >= 0).all())
            self.assertTrue((scaled_data[col] <= 1).all())

        # Verifica il risultato atteso per una colonna
        expected_A = (self.data["A"] - self.data["A"].min()) / (self.data["A"].max() - self.data["A"].min())
        pd.testing.assert_series_equal(scaled_data["A"], expected_A)

    def test_standardize(self):
        # Esegui standardizzazione
        scaled_data = FeatureScaler.standardize(self.data, self.exclude_columns)

        # Controlla che le colonne escluse rimangano invariate
        pd.testing.assert_series_equal(scaled_data["C"], self.data["C"])

        # Controlla che le colonne scalate abbiano media 0 e deviazione standard 1
        for col in ["A", "B"]:
            self.assertAlmostEqual(scaled_data[col].mean(), 0, places=6)
            self.assertAlmostEqual(scaled_data[col].std(), 1, places=6)

        # Verifica il risultato atteso per una colonna
        expected_A = (self.data["A"] - self.data["A"].mean()) / self.data["A"].std()
        pd.testing.assert_series_equal(scaled_data["A"], expected_A)

    def test_invalid_strategy(self):
        with self.assertRaises(ValueError):
            FeatureScalerStrategyManager.scale_features("invalid_strategy", self.data, self.exclude_columns)

    def test_strategy_normalize(self):
        # Usa la strategia di normalizzazione
        scaled_data = FeatureScalerStrategyManager.scale_features("normalize", self.data, self.exclude_columns)

        # Controlla che le colonne scalate siano normalizzate
        for col in ["A", "B"]:
            self.assertTrue((scaled_data[col] >= 0).all())
            self.assertTrue((scaled_data[col] <= 1).all())

    def test_strategy_standardize(self):
        # Usa la strategia di standardizzazione
        scaled_data = FeatureScalerStrategyManager.scale_features("standardize", self.data, self.exclude_columns)

        # Controlla che le colonne scalate abbiano media 0 e deviazione standard 1
        for col in ["A", "B"]:
            self.assertAlmostEqual(scaled_data[col].mean(), 0, places=6)
            self.assertAlmostEqual(scaled_data[col].std(), 1, places=6)

if __name__ == "__main__":
    unittest.main()
