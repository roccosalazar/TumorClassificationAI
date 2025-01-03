import unittest
import pandas as pd
from models.knn import KNNClassifier

class TestKNNClassifier(unittest.TestCase):
    def setUp(self):
        """
        Configura i dati di test per il KNNClassifier.
        """
        # Dataset di esempio
        self.training_data = pd.DataFrame({
            'Feature1': [1, 2, 3, 6],
            'Feature2': [1, 3, 2, 8]
        })
        self.training_labels = pd.Series([0, 1, 1, 0])

        self.test_data = pd.DataFrame({
            'Feature1': [2.5, 4],
            'Feature2': [2.5, 6]
        })

        self.expected_predictions = [1, 0]  # Risultati attesi con k=3

        self.knn = KNNClassifier(k=3)

    def test_fit(self):
        """
        Verifica che il metodo fit memorizzi correttamente i dati di training.
        """
        self.knn.fit(self.training_data, self.training_labels)
        pd.testing.assert_frame_equal(self.knn.data, self.training_data)
        pd.testing.assert_series_equal(self.knn.labels, self.training_labels)

    def test_predict(self):
        """
        Verifica che il metodo predict fornisca il risultato atteso.
        """
        self.knn.fit(self.training_data, self.training_labels)
        prediction = self.knn.predict(self.test_data.iloc[0])
        self.assertEqual(prediction, self.expected_predictions[0])

    def test_invalid_fit_input(self):
        """
        Verifica che venga sollevata un'eccezione se i dati forniti a fit non sono validi.
        """
        with self.assertRaises(ValueError):
            self.knn.fit([1, 2, 3], self.training_labels)  # Non è un DataFrame

    def test_invalid_predict_input(self):
        """
        Verifica che venga sollevata un'eccezione se il punto fornito a predict non è valido.
        """
        self.knn.fit(self.training_data, self.training_labels)
        with self.assertRaises(ValueError):
            self.knn.predict([2.5, 2.5])  # Non è un Series

    def tearDown(self):
        """
        Pulisce le risorse utilizzate nei test.
        """
        del self.knn

if __name__ == '__main__':
    unittest.main()
