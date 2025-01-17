import unittest
import pandas as pd
from validation import LeavePOutCV

class TestLeavePOutCV(unittest.TestCase):
    def setUp(self):
        """
        Prepara i dati di esempio per i test.
        """
        # Dataset di esempio
        self.data = pd.DataFrame({
            "feature1": [1, 2, 3, 4],
            "feature2": [10, 20, 30, 40]
        })
        self.labels = pd.Series([1, 0, 1, 0])

    def test_generate_splits_valid(self):
        """
        Testa il funzionamento di base di generate_splits con p=2.
        """
        leave_p_out = LeavePOutCV(p=2, n_combinations=10)  # Imposta n_combinations
        results = leave_p_out.generate_splits(self.data, self.labels)

        # Verifica che il risultato sia una lista
        self.assertIsInstance(results, list)
        # Verifica che il numero di combinazioni sia uguale a n_combinations
        self.assertEqual(len(results), 10)

        # Verifica che ogni elemento sia una tupla
        for y_real, y_pred in results:
            self.assertIsInstance(y_real, list)
            self.assertIsInstance(y_pred, list)

    def test_generate_splits_small_p(self):
        """
        Testa generate_splits con p=1 (Leave-One-Out) e un numero fisso di combinazioni.
        """
        leave_p_out = LeavePOutCV(p=1, n_combinations=4)  # Imposta n_combinations a 4
        results = leave_p_out.generate_splits(self.data, self.labels)

        # Verifica che il numero di combinazioni sia uguale a n_combinations
        self.assertEqual(len(results), 4)

    def test_generate_splits_large_p(self):
        """
        Testa generate_splits con p uguale al numero di campioni.
        """
        leave_p_out = LeavePOutCV(p=4, n_combinations=10)  # Numero totale di campioni = 4
        with self.assertRaises(ValueError):
            leave_p_out.generate_splits(self.data, self.labels)

    def test_generate_splits_invalid_p(self):
        """
        Testa generate_splits con un valore di p non valido.
        """
        with self.assertRaises(ValueError):
            LeavePOutCV(p=0)  # p deve essere positivo

        with self.assertRaises(ValueError):
            LeavePOutCV(p=-1)  # p non può essere negativo

    def test_generate_splits_empty_dataset(self):
        """
        Testa generate_splits con un dataset vuoto.
        """
        data = pd.DataFrame(columns=["feature1", "feature2"])
        labels = pd.Series(dtype=int)

        leave_p_out = LeavePOutCV(p=1, n_combinations=10)
        results = leave_p_out.generate_splits(data, labels)

        # Il risultato deve essere una lista vuota
        self.assertEqual(len(results), 0)

if __name__ == "__main__":
    unittest.main()
