from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from knn import KNN  # Importa la classe KNN

# Interfaccia Validation
class Validation(ABC):
    @abstractmethod
    def validate(self, data: pd.DataFrame, target_column: str):
        pass

# Classe Holdout che implementa Validation
class Holdout(Validation):
    def __init__(self, test_size=0.2, random_state=None, k=3):
        """
        Inizializza i parametri per la validazione Holdout.

        Args:
            test_size (float): Proporzione del dataset da usare per il testing.
            random_state (int): Seed per la riproducibilità.
            k (int): Numero di vicini per il classificatore k-NN.
        """
        self.test_size = test_size
        self.random_state = random_state
        self.k = k

    def validate(self, data: pd.DataFrame, target_column: str):
        """
        Esegue la validazione Holdout.

        Args:
            data (pd.DataFrame): Il dataset completo.
            target_column (str): Il nome della colonna target.

        Returns:
            dict: Metriche di valutazione.
        """
        # Separazione delle feature e della target
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Shuffle del dataset
        if self.random_state is not None:
            np.random.seed(self.random_state)
        shuffled_indices = np.random.permutation(len(data))

        # Suddivisione in training e testing
        test_size = int(len(data) * self.test_size)
        test_indices = shuffled_indices[:test_size]
        train_indices = shuffled_indices[test_size:]

        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

        # Inizializza e addestra il classificatore k-NN
        knn = KNN(k=self.k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test).to_numpy().flatten()

        # Calcolo delle metriche di valutazione
        return self.calculate_metrics(y_test.values, y_pred)

    def calculate_metrics(self, y_true, y_pred):
        """
        Calcola metriche di valutazione senza librerie esterne.

        Args:
            y_true (np.ndarray): Etichette reali.
            y_pred (np.ndarray): Etichette predette.

        Returns:
            dict: Metriche calcolate.
        """
        metrics = {}
        metrics['Accuracy'] = (y_true == y_pred).sum() / len(y_true)
        tp = ((y_true == 4) & (y_pred == 4)).sum()
        tn = ((y_true == 2) & (y_pred == 2)).sum()
        fp = ((y_true == 2) & (y_pred == 4)).sum()
        fn = ((y_true == 4) & (y_pred == 2)).sum()
        metrics['Confusion Matrix'] = [[tn, fp], [fn, tp]]
        if tp + fn > 0 and tn + fp > 0:
            metrics['Sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            metrics['Sensitivity'] = 'N/A'
            metrics['Specificity'] = 'N/A'
        return metrics

# Esempio di utilizzo con un dataset simulato
data = pd.DataFrame({
    'Feature1': np.random.rand(100),
    'Feature2': np.random.rand(100),
    'Class': np.random.choice([2, 4], size=100)
})

# Creazione di un'istanza di Holdout
holdout_validator = Holdout(test_size=0.2, random_state=42, k=3)
metrics = holdout_validator.validate(data, target_column='Class')

print("Metriche di valutazione:", metrics)
print("cambiamento")