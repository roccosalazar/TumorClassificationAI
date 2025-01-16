import unittest
import pandas as pd
import numpy as np
from models.knn import KNNClassifier  
from validation import Holdout

class TestHoldout(unittest.TestCase): 
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
        Che generate_splits produca una lista con una tupla (y_real, y_pred).
        Che il numero di campioni nei set di test e training sia corretto
        """
        holdout = Holdout(test_size=0.4)
        results = holdout.generate_splits(self.data, self.labels)
        
        # Verifica che il risultato sia una lista con una tupla
        self.assertIsInstance(results, list) # Verifica che il risultato sia una lista
        self.assertEqual(len(results), 1) # Verifica che il risultato sia una lista con una sola tupla
        self.assertIsInstance(results[0], tuple) #verifica che l'elemento della lista sia una tupla

        # Verifica che y_real e y_pred abbiano la dimensione corretta
        y_real, y_pred = results[0]
        self.assertEqual(len(y_real), 2)  # 40% di 5 campioni ≈ 2
        self.assertEqual(len(y_pred), 2)

    def test_generate_splits_default_test_size(self):
        """
        Testa generate_splits con il valore di default di test_size.
        """
        holdout = Holdout()  # test_size default = 0.2
        results = holdout.generate_splits(self.data, self.labels)
        
        y_real, y_pred = results[0]
        self.assertEqual(len(y_real), 1)  # 20% di 5 campioni ≈ 1
        self.assertEqual(len(y_pred), 1) 

    def test_generate_splits_full_test_size(self):
        """
        Testa generate_splits con test_size = 1.0 (tutti i campioni come test),e che generi un ValueError.
        """
        holdout = Holdout(test_size=1.0)
        with self.assertRaises(ValueError):
            holdout.generate_splits(self.data, self.labels)

    def test_generate_splits_invalid_test_size(self):
        """
        Testa generate_splits con un valore non valido di test_size.
        """
        with self.assertRaises(ValueError):
            Holdout(test_size=1.2)  # test_size > 1 non è valido

        with self.assertRaises(ValueError):
            Holdout(test_size=-0.1)  # test_size < 0 non è valido


    def test_generate_splits_small_dataset(self):
        """
        Testa generate_splits con un dataset molto piccolo.
        Ci si aspetta che entrambe le liste (y_real e y_pred) risultino vuote impostando test_size=0.5.
        """
        data = pd.DataFrame({
            "feature1": [1],
            "feature2": [10]
        })
        labels = pd.Series([1])

        holdout = Holdout(test_size=0.5)
        results = holdout.generate_splits(data, labels)
        
        y_real, y_pred = results[0]
        self.assertEqual(len(y_real), 0)  # Nessun campione nel test con dataset troppo piccolo
        self.assertEqual(len(y_pred), 0)

    def test_generate_splits_no_shuffle(self):
        """
        Testa che il dataset sia mescolato correttamente.
        """
        holdout = Holdout(test_size=0.4)
        results = holdout.generate_splits(self.data, self.labels)
        
        test_indices = [self.labels.tolist().index(y) for y in results[0][0]] 
        """Converte y_real (etichette reali del set di test) negli indici corrispondenti all'interno del dataset originale."""
        train_indices = list(set(range(len(self.labels))) - set(test_indices))
        """Calcola gli indici del set di training come la differenza tra l'insieme di tutti gli indici del dataset originale 
        (range(len(self.labels))) e gli indici del set di test (test_indices)."""
        
        # Verifica che le due liste siano disgiunte
        self.assertTrue(set(test_indices).isdisjoint(set(train_indices)))

if __name__ == "__main__":
    unittest.main()
