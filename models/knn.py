import pandas as pd
import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3):
        """
        Inizializza il classificatore KNN con il numero di vicini (k).
        """
        self.k = k
        self.data = None
        self.labels = None

    def fit(self, data: pd.DataFrame, labels: pd.Series):
        """
        Memorizza i dati di training e le rispettive etichette.

        Args:
            data (pd.DataFrame): Dataset di training con le feature.
            labels (pd.Series): Etichette corrispondenti ai dati di training.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("I dati devono essere forniti come Pandas DataFrame.")
        if not isinstance(labels, pd.Series):
            raise ValueError("Le etichette devono essere fornite come Pandas Series.")
        
        self.data = data
        self.labels = labels

    def predict(self, point: pd.Series) -> any:
        """
        Classifica un singolo punto utilizzando i dati di training.

        Args:
            point (pd.Series): Il punto da classificare.

        Returns:
            Etichetta predetta.
        """
        if self.data is None or self.labels is None:
            raise ValueError("Il modello non è stato addestrato. Usa il metodo 'fit' prima di predire.")
        
        if not isinstance(point, pd.Series):
            raise ValueError("Il punto deve essere un Pandas Series.")
        
        # Calcolo delle distanze euclidee
        distances = np.sqrt(((self.data - point) ** 2).sum(axis=1))
        
        # Selezione dei k vicini più vicini
        nearest_neighbors = distances.nsmallest(self.k).index
        
        # Determinazione della classe più frequente
        nearest_labels = self.labels.loc[nearest_neighbors]
        label_count = Counter(nearest_labels)
        most_common = label_count.most_common(1)
        return most_common[0][0]

    def predict_batch(self, points: pd.DataFrame) -> pd.Series:
        """
        Classifica un batch di punti utilizzando i dati di training.

        Args:
            points (pd.DataFrame): Dataset di punti da classificare.

        Returns:
            pd.Series: Etichette predette per ogni punto.
        """
        if not isinstance(points, pd.DataFrame):
            raise ValueError("I punti devono essere forniti come Pandas DataFrame.")
        
        predictions = points.apply(self.predict, axis=1)
        return predictions
