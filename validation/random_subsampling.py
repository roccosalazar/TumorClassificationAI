import numpy as np
import pandas as pd
from models.knn import KNNClassifier
from .validation_strategy import ValidationStrategy

class RandomSubsampling(ValidationStrategy):
    def __init__(self, n_iter=10, test_size=0.2):
        """
        Inizializza la strategia Random Subsampling.

        Args:
            n_iter (int): Numero di iterazioni di subsampling (default 10).
            test_size (float): Percentuale del dataset da utilizzare come test (default 0.2).
        """
        if not isinstance(n_iter, int) or n_iter <= 0:
            raise ValueError("Il numero di iterazioni (n_iter) deve essere un intero positivo.")
        if not (0 < test_size <= 1):
            raise ValueError("test_size deve essere compreso tra 0 e 1.")
        
        self.n_iter = n_iter
        self.test_size = test_size

    def generate_splits(self, data: pd.DataFrame, labels: pd.Series, k=3) -> list[tuple[list[int], list[int]]]:
        """
        Genera più divisioni randomiche del dataset e restituisce una lista di tuple.

        Args:
            data (pd.DataFrame): Le feature del dataset.
            labels (pd.Series): Le etichette del dataset.
            k (int): Numero di vicini per il KNN (default 3).

        Returns:
            list[tuple[list[int], list[int]]]: Lista di tuple (y_real, y_pred).
        """
        results = []
        for _ in range(self.n_iter):
            n_samples = len(data)
            n_test = int(n_samples * self.test_size)
            
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
            
            # Aggiungi la tupla (y_real, y_pred) alla lista dei risultati
            results.append((test_labels.tolist(), predictions.tolist()))
        
        return results
