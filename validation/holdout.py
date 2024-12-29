from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from validation import Validation 


# Classe Holdout che implementa Validation
class Holdout(Validation):
    def __init__(self, test_size=0.2, random_state=None):
        """
        Inizializza i parametri per la validazione Holdout.

        Args:
            test_size (float): Proporzione del dataset da usare per il testing.
            random_state (int): Seed per la riproducibilità.
        """
        self.test_size = test_size
        self.random_state = random_state

    def split(self, data: pd.DataFrame):
        """
        Divide il dataset in training e test set secondo la validazione Holdout.

        Args:
            data (pd.DataFrame): Dataset da dividere.

        Returns:
            folds: Lista contenente una tupla (training set, test set).
        """
        # Lista dei folds
        folds = []

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Permutazione casuale degli indici
        shuffled_indices = np.random.permutation(len(data))
        
        # Calcolo della dimensione del test set
        test_set_size = int(len(data) * self.test_size)
        
        # Divisione degli indici
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]

        # Creazione dei set e aggiunta alla lista
        folds.append((data.iloc[train_indices], data.iloc[test_indices]))
        return folds
