import numpy as np
import pandas as pd
from collections import Counter

class KNN:
    def __init__(self, k=3):
        """
        Inizializza il classificatore k-NN con il numero di vicini.

        Args:
            k (int): Numero di vicini da considerare per la classificazione.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Memorizza i dati di training.

        Args:
            X_train (pd.DataFrame): Caratteristiche del set di training.
            y_train (pd.Series): Etichette del set di training.
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Prevede le etichette per il set di test.

        Args:
            X_test (pd.DataFrame): Caratteristiche del set di test.

        Returns:
            pd.DataFrame: Etichette previste per il set di test con indice originale.
        """
        predictions = [self._predict_single_point(x) for x in X_test.to_numpy()]
        return pd.DataFrame(predictions, index=X_test.index, columns=["Prediction"])

    def _predict_single_point(self, x: np.ndarray):
        """
        Prevede l'etichetta per un singolo punto.

        Args:
            x (np.ndarray): Un singolo campione del set di test.

        Returns:
            int/float/str: L'etichetta prevista per il campione.
        """
        # Calcola la distanza euclidea tra il punto e tutti i punti del training
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train.to_numpy()]
        
        # Trova gli indici dei k vicini più vicini
        k_indices = np.argsort(distances)[:self.k]
        
        # Ottieni le etichette dei k vicini
        k_nearest_labels = [self.y_train.iloc[i] for i in k_indices]

        # Determina l'etichetta più comune
        label_counter = Counter(k_nearest_labels)
        most_common = label_counter.most_common(1)

        # Gestione del pareggio
        max_count = most_common[0][1]
        tied_labels = [label for label, count in label_counter.items() if count == max_count]
        
        if len(tied_labels) > 1:
            return np.random.choice(tied_labels)
        else:
            return most_common[0][0]

    @staticmethod
    def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calcola la distanza euclidea tra due punti.

        Args:
            x1 (np.ndarray): Primo punto.
            x2 (np.ndarray): Secondo punto.

        Returns:
            float: Distanza euclidea.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

# Esempio di utilizzo della classe
if __name__ == "__main__":
    # Creazione di un DataFrame di esempio
    df_train = pd.DataFrame({
        "feature1": [1, 2, 3, 6],
        "feature2": [2, 3, 4, 7],
        "classtype_v1": [2, 2, 4, 4]
    }, index=["Sample1", "Sample2", "Sample3", "Sample4"])

    df_test = pd.DataFrame({
        "feature1": [1.5, 5],
        "feature2": [2.5, 5]
    }, index=["Sample5", "Sample6"])

    # Separazione delle caratteristiche e delle etichette
    X_train = df_train[["feature1", "feature2"]]
    y_train = df_train["classtype_v1"]

    # Crea e addestra il modello KNN
    knn = KNN(k=3)
    knn.fit(X_train, y_train)

    # Prevede le etichette
    predictions = knn.predict(df_test)
    print("Predictions:\n", predictions)
