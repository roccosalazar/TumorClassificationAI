import numpy as np
import unittest
from validation.metrics_calculator import MetricsCalculator


class TestMetricsCalculator(unittest.TestCase):
    def test_holdout_accuracy(self):
        y_real = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 0])
        input_data = [(y_real, y_pred)]
        calculator = MetricsCalculator(input_data, method="holdout")
        metrics = calculator.calculate_metrics()
        self.assertAlmostEqual(metrics["Accuracy"], 0.8)

    def test_holdout_error_rate(self):
        y_real = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 0])
        input_data = [(y_real, y_pred)]
        calculator = MetricsCalculator(input_data, method="holdout")
        metrics = calculator.calculate_metrics()
        self.assertAlmostEqual(metrics["Error Rate"], 0.2)

    def test_holdout_sensitivity(self):
        y_real = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 0])
        input_data = [(y_real, y_pred)]
        calculator = MetricsCalculator(input_data, method="holdout")
        metrics = calculator.calculate_metrics()
        self.assertAlmostEqual(metrics["Sensitivity"], 0.6666666666666666)

    def test_LeavePoutCrossValidation_aggregation(self):
    # Genera una matrice 20x2 con coppie di liste di etichette di lunghezza 10
        np.random.seed(42)
        input_data = [
        (np.random.randint(0, 2, size=10).tolist(), np.random.randint(0, 2, size=10).tolist())
        for _ in range(20)
        ]

    # Inizializza il calcolatore con metodo Leave-p-out Cross Validation
        calculator = MetricsCalculator(input_data, method="Leave-p-out Cross Validation")

    # Calcola le metriche
        metrics = calculator.calculate_metrics()

    # Stampa le metriche calcolate
        print("Metriche Leave-p-out Cross Validation (20x2, liste di etichette lunghe 10):")
        print(metrics)

    # Verifica valori calcolati (adatta secondo i risultati attesi)
    # Nota: qui Accuracy, Error Rate, e Sensitivity sono placeholder e dipendono dall'input generato.
        #self.assertAlmostEqual(metrics["Accuracy"], 0.5, places=2)  # Cambia con il valore atteso
        #self.assertAlmostEqual(metrics["Error Rate"], 0.5, places=2)  # Cambia con il valore atteso
        #self.assertAlmostEqual(metrics["Sensitivity"], 0.5, places=2)  # Cambia con il valore atteso


        if __name__ == "__main__":
         unittest.main()