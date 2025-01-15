import numpy as np
import pandas as pd
from models.knn import KNNClassifier
from .validation_strategy import ValidationStrategy

class Holdout(ValidationStrategy):
    def __init__(self, test_size=0.2):
        """
        Inizializza la strategia Holdout con una dimensione del set di test.

        Args:
            test_size (float): Percentuale del dataset da utilizzare come test (default 0.2).
        """
        if not (0 < test_size <= 1):
            raise ValueError("test_size deve essere compreso tra 0 e 1.")
        self.test_size = test_size

    def generate_splits(self, data: pd.DataFrame, labels: pd.Series, k=3) -> list[tuple[list[int], list[int]]]:
        n_samples = len(data)
        n_test = int(n_samples * self.test_size)
        if n_test == n_samples:
            raise ValueError("Il set di training è vuoto. Riduci il valore di test_size.")
        
        # Shuffle del dataset
        shuffled_indices = np.random.permutation(n_samples)
        test_indices = shuffled_indices[:n_test]
        train_indices = shuffled_indices[n_test:]
        
        # Split
        train_data, test_data = data.iloc[train_indices], data.iloc[test_indices]
        train_labels, test_labels = labels.iloc[train_indices], labels.iloc[test_indices]
        
        # Addestramento e predizione
        knn = KNNClassifier(k)
        knn.fit(train_data, train_labels)
        predictions = knn.predict_batch(test_data)
        
        # Costruisce la lista di tuple (y_real, y_pred)
        results = [(test_labels.tolist(), predictions.tolist())]

        """Il primo elemento (test_labels.tolist()) è una lista delle etichette reali del set di test.
        Il secondo elemento (predictions.tolist()) è una lista delle etichette previste dal modello."""
        
        return results
