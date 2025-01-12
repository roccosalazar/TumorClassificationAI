import numpy as np
import pandas as pd
from models.knn import KNNClassifier
from .validation_strategy import ValidationStrategy

class LeavePOutCV(ValidationStrategy):
    def __init__(self, p=2, n_combinations=100):
        """
        Inizializza la strategia Leave-P-Out Cross Validation con combinazioni casuali.

        Args:
            p (int): Numero di campioni da lasciare fuori ad ogni iterazione.
            n_combinations (int): Numero di combinazioni casuali da generare.
        """
        if p <= 0:
            raise ValueError("Il valore di 'p' deve essere positivo.")
        if n_combinations <= 0:
            raise ValueError("Il valore di 'n_combinations' deve essere positivo.")
        self.p = p
        self.n_combinations = n_combinations

    def generate_splits(self, data: pd.DataFrame, labels: pd.Series, k=3) -> list[tuple[list[int], list[int]]]:
        """
        Genera combinazioni Leave-P-Out casuali e restituisce una lista di tuple.

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

        results = []
        rng = np.random.default_rng()  # Generatore di numeri casuali

        for _ in range(self.n_combinations):
            # Genera una combinazione casuale di 'p' indici
            test_indices = rng.choice(n_samples, self.p, replace=False)
            train_indices = np.setdiff1d(np.arange(n_samples), test_indices)

            # Split
            train_data, test_data = data.iloc[train_indices], data.iloc[test_indices]
            train_labels, test_labels = labels.iloc[train_indices], labels.iloc[test_indices]

            # Addestramento e predizione
            knn = KNNClassifier(k)
            if train_data.empty:  # Verifica che il training set non sia vuoto
                raise ValueError("Il training set è vuoto. Riduci il valore di 'p'.")
            knn.fit(train_data, train_labels)
            predictions = knn.predict_batch(test_data)

            # Aggiungi la tupla (y_real, y_pred) alla lista dei risultati
            results.append((test_labels.tolist(), predictions.tolist()))

        return results
