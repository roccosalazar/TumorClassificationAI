import pandas as pd
import numpy as np
from .validation import Validation

class KFoldValidation(Validation):
    def __init__(self, k=5, random_state=None):
        self.k = k
        self.random_state = random_state

    def split(self, data: pd.DataFrame, target_column: str):
        """
        Suddivide il dataset in K fold per cross-validation.
        """
        if self.random_state:
            np.random.seed(self.random_state)

        indices = np.random.permutation(len(data))
        fold_size = len(data) // self.k
        folds = []
        
        for i in range(self.k):
            test_indices = indices[i * fold_size:(i + 1) * fold_size]
            train_indices = np.setdiff1d(indices, test_indices)
            train_data = data.iloc[train_indices]
            test_data = data.iloc[test_indices]
            folds.append((train_data, test_data))
        
        return folds

    def evaluate(self, model, data: pd.DataFrame, target_column: str):
        """
        Valuta il modello usando K-Fold Cross-Validation.
        """
        folds = self.split(data, target_column)
        accuracies = []
        
        for train_data, test_data in folds:
            model.fit(train_data.drop(columns=[target_column]), train_data[target_column])
            predictions = model.predict_batch(test_data.drop(columns=[target_column]))
            accuracy = (predictions == test_data[target_column]).mean()
            accuracies.append(accuracy)
        
        return np.mean(accuracies)
