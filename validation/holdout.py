import pandas as pd
import numpy as np
from knn import KNN  # Importa la classe KNN

def holdout_validation(data, target_column, test_size=0.2, random_state=None, k=3):
    """
    Implementa la validazione Holdout per dividere il dataset in training e testing,
    addestra un classificatore e calcola le metriche di valutazione.

    Args:
        data (pd.DataFrame): Il dataset completo.
        target_column (str): Il nome della colonna target.
        test_size (float): La proporzione del dataset da usare per il testing.
        random_state (int): Seed per la riproducibilità.
        k (int): Numero di vicini per il classificatore k-NN.

    Returns:
        dict: Un dizionario contenente le metriche di valutazione.
    """
    # Separazione delle feature e della target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Shuffle del dataset
    if random_state is not None:
        np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(data))

    # Suddivisione in training e testing
    test_size = int(len(data) * test_size)
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]

    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    # Inizializza e addestra il classificatore k-NN
    knn = KNN(k=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test).to_numpy().flatten()

    # Calcolo delle metriche di valutazione
    def calculate_metrics(y_true, y_pred):
        """
        Calcola metriche di valutazione senza librerie esterne.
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

    metrics = calculate_metrics(y_test.values, y_pred)

    return metrics

# Esempio di utilizzo con un dataset simulato
data = pd.DataFrame({
    'Feature1': np.random.rand(100),
    'Feature2': np.random.rand(100),
    'Class': np.random.choice([2, 4], size=100)
})

metrics = holdout_validation(data, target_column='Class', test_size=0.2, random_state=42, k=3)
print("Metriche di valutazione:", metrics)
