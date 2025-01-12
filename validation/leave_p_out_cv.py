import numpy as np
import pandas as pd
from itertools import combinations
from models.knn import KNNClassifier
from .validation_strategy import ValidationStrategy

class LeavePOutCV(ValidationStrategy):
    def __init__(self, p=2):
        """
        Inizializza la strategia Leave-P-Out Cross Validation.

        Args:
            p (int): Numero di campioni da lasciare fuori ad ogni iterazione.
        """
        if p <= 0:
            raise ValueError("Il valore di 'p' deve essere positivo.")
        self.p = p

    def generate_splits(self, data: pd.DataFrame, labels: pd.Series, k=3) -> list[tuple[list[int], list[int]]]:
        """
        Genera tutte le combinazioni Leave-P-Out e restituisce una lista di tuple.

        Args:
            data (pd.DataFrame): Le feature del dataset.
            labels (pd.Series): Le etichette del dataset.
            k (int): Numero di vicini per il KNN (default 3).

        Returns:
            list[tuple[list[int], list[int]]]: Lista di tuple (y_real, y_pred).
        """
        n_samples = len(data)
        
        # Se il dataset è vuoto, restituiamo direttamente una lista vuota
        if n_samples == 0:
            return []

        if self.p > n_samples:
            raise ValueError("Il valore di 'p' non può essere maggiore del numero totale di campioni nel dataset.")

        indices = np.arange(n_samples)
        results = []

        for test_indices in combinations(indices, self.p):
            train_indices = np.setdiff1d(indices, test_indices)
            
            # Split
            train_data, test_data = data.iloc[train_indices], data.iloc[list(test_indices)]
            train_labels, test_labels = labels.iloc[train_indices], labels.iloc[list(test_indices)]
            
            # Addestramento e predizione
            knn = KNNClassifier(k)
            if train_data.empty:  # Verifica che il training set non sia vuoto
                raise ValueError("Il training set è vuoto. Riduci il valore di 'p'.")
            knn.fit(train_data, train_labels)
            predictions = knn.predict_batch(test_data)
            
            # Aggiungi la tupla (y_real, y_pred) alla lista dei risultati
            results.append((test_labels.tolist(), predictions.tolist()))
        
        return results

