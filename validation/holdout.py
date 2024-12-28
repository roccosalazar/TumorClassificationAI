from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


# Interfaccia Validation
class Validation(ABC):
    @abstractmethod
    def validate(self, data: pd.DataFrame, target_column: str):
        pass

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
        Il metodo split esegue la holdout sul dataset
        :pd.DataFrame: dataframe da dividere in set di training e test
        :return: folds: lista di tuple, ognuna delle quali contiene il set di training e il set di test (in questo caso una sola tupla con il set di training e il set di test)
        """
         # Lista dei folds
        folds = []

        if self.random_state is not None:
            np.random.seed(self.random_state)  # inizializzazione del generatore di numeri casuali

        shuffled_indices = np.random.permutation(len(df))  # permutazione casuale degli indici del DataFrame
        test_set_size = int(
            len(df) * self.test_size)  # arrotondamento per difetto all'intero più vicino della dimensione del test set
        test_indices = shuffled_indices[:test_set_size]  # selezione degli indici del test set
        train_indices = shuffled_indices[test_set_size:]  # selezione degli indici del training set

        folds.append((df.iloc[train_indices], df.iloc[test_indices]))  # accesso alle righe tramite gli indici di posizione (iloc) selezionati e aggiunta delle tuple alla lista dei folds
        return folds