import unittest
from metrics import MetricsCalculator

class TestMetricsCalculator(unittest.TestCase):

    def setUp(self):
        # Setup iniziale: crea un'istanza della classe MetricsCalculator e fornisci alcuni dati di esempio
        self.calculator = MetricsCalculator()
        self.sample_data = [
            ([1, 0, 1, 1, 0], [1, 0, 1, 0, 0]),
            ([0, 1, 1, 0, 0], [0, 1, 0, 0, 1]),
            ([1, 1, 0, 1, 0], [1, 1, 0, 0, 0]),
            ([0, 0, 1, 1, 1], [0, 0, 1, 1, 0]),
        ]

    def test_accuracy_rate(self):
        for y_true, y_pred in self.sample_data:
            tp, tn, fp, fn = self.calculator._confusion_matrix(y_true, y_pred)
            accuracy_rate = self.calculator._accuracy_rate(tp, tn, fp, fn)
            # Calcolo manuale dell'accuracy per confronto
            total = tp + tn + fp + fn
            expected_accuracy = (tp + tn) / total if total > 0 else 0.0
            self.assertAlmostEqual(accuracy_rate, expected_accuracy)

    def test_error_rate(self):
        for y_true, y_pred in self.sample_data:
            tp, tn, fp, fn = self.calculator._confusion_matrix(y_true, y_pred)
            error_rate = self.calculator._error_rate(tp, tn, fp, fn)
            # Calcolo manuale dell'error rate per confronto
            total = tp + tn + fp + fn
            expected_error_rate = (fp + fn) / total if total > 0 else 0.0
            self.assertAlmostEqual(error_rate, expected_error_rate)

    def test_sensitivity(self):
        for y_true, y_pred in self.sample_data:
            tp, tn, fp, fn = self.calculator._confusion_matrix(y_true, y_pred)
            sensitivity = self.calculator._sensitivity(tp, fn)
            # Calcolo manuale della sensitivity per confronto
            actual_positive = tp + fn
            expected_sensitivity = tp / actual_positive if actual_positive > 0 else 0.0
            self.assertAlmostEqual(sensitivity, expected_sensitivity)

    def test_specificity(self):
        for y_true, y_pred in self.sample_data:
            tp, tn, fp, fn = self.calculator._confusion_matrix(y_true, y_pred)
            specificity = self.calculator._specificity(tn, fp)
            # Calcolo manuale della specificity per confronto
            actual_negative = tn + fp
            expected_specificity = tn / actual_negative if actual_negative > 0 else 0.0
            self.assertAlmostEqual(specificity, expected_specificity)

    def test_geometric_mean(self):
        for y_true, y_pred in self.sample_data:
            tp, tn, fp, fn = self.calculator._confusion_matrix(y_true, y_pred)
            g_mean = self.calculator._geometric_mean(tp, tn, fp, fn)
            # Calcolo manuale della geometric mean per confronto
            sensitivity = self.calculator._sensitivity(tp, fn)
            specificity = self.calculator._specificity(tn, fp)
            expected_g_mean = (sensitivity * specificity) ** 0.5
            self.assertAlmostEqual(g_mean, expected_g_mean)

    def test_area_under_curve(self):
        for y_true, y_pred in self.sample_data:
            tp, tn, fp, fn = self.calculator._confusion_matrix(y_true, y_pred)
            auc = self.calculator._area_under_curve(tp, tn, fp, fn)
            # Calcolo manuale dell'area under curve per confronto
            sensitivity = self.calculator._sensitivity(tp, fn)
            specificity = self.calculator._specificity(tn, fp)
            expected_auc = (sensitivity + specificity) / 2
            self.assertAlmostEqual(auc, expected_auc)

if __name__ == '__main__':
    unittest.main() 
