import numpy as np
import pandas as pd
from .validation import Validation


class HoldoutValidation(Validation):
    def __init__(self, test_size=0.2, random_state=None):
        self.test_size = test_size
        self.random_state = random_state

    def split(self, data: pd.DataFrame, target_column: str):
        """
        Suddivide il dataset in training e test set.
        """
        if self.random_state:
            np.random.seed(self.random_state)
        
        indices = np.random.permutation(len(data))
        test_count = int(len(data) * self.test_size)
        test_indices = indices[:test_count]
        train_indices = indices[test_count:]
        
        return data.iloc[train_indices], data.iloc[test_indices]

    def evaluate(self, model, data: pd.DataFrame, target_column: str):
        """
        Valuta il modello usando Holdout.
        """
        train_data, test_data = self.split(data, target_column)
        model.fit(train_data.drop(columns=[target_column]), train_data[target_column])
        predictions = model.predict_batch(test_data.drop(columns=[target_column]))
        accuracy = (predictions == test_data[target_column]).mean()
        return accuracy
