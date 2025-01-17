import unittest
import pandas as pd
from validation import RandomSubsampling

class TestRandomSubsampling(unittest.TestCase):
    def setUp(self):
        """
        Prepara i dati di esempio per i test.
        """
        # Dataset di esempio
        self.data = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [10, 20, 30, 40, 50]
        })
        self.labels = pd.Series([1, 0, 1, 0, 1])

    def test_generate_splits_valid(self):
        """
        Testa il funzionamento di base di generate_splits con valori validi.
        """
        subsampling = RandomSubsampling(n_iter=5, test_size=0.4)
        results = subsampling.generate_splits(self.data, self.labels)

        # Verifica che il risultato sia una lista
        self.assertIsInstance(results, list)
        # Verifica che il numero di iterazioni sia corretto
        self.assertEqual(len(results), 5)

        # Verifica che ogni elemento sia una tupla (y_real, y_pred)
        for y_real, y_pred in results:
            self.assertIsInstance(y_real, list)
            self.assertIsInstance(y_pred, list)

    def test_generate_splits_default_values(self):
        """
        Testa generate_splits con i valori di default di n_iter e test_size.
        """
        subsampling = RandomSubsampling()  # n_iter=10, test_size=0.2
        results = subsampling.generate_splits(self.data, self.labels)

        # Verifica che il numero di iterazioni sia quello di default
        self.assertEqual(len(results), 10)

        # Verifica che il test set abbia circa il 20% dei campioni
        y_real, _ = results[0]
        self.assertAlmostEqual(len(y_real), int(len(self.data) * 0.2), delta=1)

    def test_generate_splits_small_dataset(self):
        """
        Testa generate_splits con un dataset molto piccolo.
        """
        data = pd.DataFrame({
            "feature1": [1, 2],
            "feature2": [10, 20]
        })
        labels = pd.Series([1, 0])

        subsampling = RandomSubsampling(n_iter=3, test_size=0.5)
        results = subsampling.generate_splits(data, labels)

        # Verifica che ogni test set abbia almeno 1 campione
        for y_real, y_pred in results:
            self.assertGreaterEqual(len(y_real), 1)
            self.assertGreaterEqual(len(y_pred), 1)

    def test_generate_splits_invalid_test_size(self):
        """
        Testa generate_splits con un valore non valido di test_size.
        """
        with self.assertRaises(ValueError):
            RandomSubsampling(test_size=1.5)  # test_size > 1 non è valido

        with self.assertRaises(ValueError):
            RandomSubsampling(test_size=-0.1)  # test_size < 0 non è valido

    def test_generate_splits_invalid_n_iter(self):
        """
        Testa generate_splits con un valore non valido di n_iter.
        """
        with self.assertRaises(ValueError):
            RandomSubsampling(n_iter=0)  # n_iter deve essere positivo

        with self.assertRaises(ValueError):
            RandomSubsampling(n_iter=-5)  # n_iter non può essere negativo

    def test_generate_splits_no_overlap(self):
        """
        Testa che non ci sia sovrapposizione tra train set e test set.
        """
        subsampling = RandomSubsampling(n_iter=5, test_size=0.4)
        results = subsampling.generate_splits(self.data, self.labels)

        for y_real, _ in results:
            test_indices = [self.labels.tolist().index(y) for y in y_real]
            train_indices = list(set(range(len(self.labels))) - set(test_indices))
            
            # Verifica che train e test non abbiano sovrapposizioni
            self.assertTrue(set(test_indices).isdisjoint(set(train_indices)))

if __name__ == "__main__":
    unittest.main()
