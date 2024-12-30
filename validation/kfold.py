from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


# Classe astratta Validation
class Validation(ABC):
    @abstractmethod
    def split(self, data: pd.DataFrame):
        """
        Metodo astratto per suddividere il dataset in K fold.
        """
        pass


# Classe KFold che implementa Validation
class KFold(Validation):
    def __init__(self, n_splits, random_state=None):
        """
        Inizializza i parametri per la validazione K-Fold.

        Args:
            n_splits (int): Numero di fold (divisioni) del dataset.
            random_state (int): Seed per la riproducibilità.
        """
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, data: pd.DataFrame):
        """
        Divide il dataset in K fold.

        Args:
            data (pd.DataFrame): Dataset da dividere.

        Returns:
            list: Lista contenente K tuple (training set, test set).
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        shuffled_indices = np.random.permutation(len(data))  # Mescola gli indici del dataset

        fold_size = len(data) // self.n_splits  # Calcola la dimensione di ogni fold
        folds = []  # Lista per contenere i fold

        for i in range(self.n_splits):
            # Indici del test set per il fold corrente
            test_indices = shuffled_indices[i * fold_size: (i + 1) * fold_size]
            # Indici del training set (tutti gli altri indici)
            train_indices = np.setdiff1d(shuffled_indices, test_indices)

            # Crea i dataset di training e test
            train = data.iloc[train_indices]
            test = data.iloc[test_indices]

            # Aggiungi la tupla (train, test) alla lista dei fold
            folds.append((train, test))

        return folds


if __name__ == "__main__":
    # Interfaccia per configurare il metodo K-Fold
    print("Configura la validazione K-Fold:")

    # Input per il numero di fold
    while True:
        try:
            n_splits = input("Inserisci il numero di fold (K) (min 2, default 5): ")
            if not n_splits:  # Se l'utente non inserisce nulla
                n_splits = 5  # Assegna il valore di default
            else:
                n_splits = int(n_splits)
                if n_splits < 2:
                    raise ValueError("Il numero di fold deve essere almeno 2.")
            break  # Esci dal ciclo se il valore è valido
        except ValueError as e:
            print(f"Errore: {e}. Riprova.")

    # Input per il random_state
    while True:
        try:
            random_state = input("Inserisci il seed per il mescolamento (default nessun seed): ")
            if not random_state:  # Se l'utente non inserisce nulla
                random_state = None  # Assegna il valore di default
            else:
                random_state = int(random_state)
            break  # Esci dal ciclo se il valore è valido
        except ValueError:
            print("Errore: Il seed deve essere un numero intero. Riprova.")

    print(f"Configurazione scelta: K={n_splits}, random_state={random_state}")

    # Dataset simulato
    data = pd.DataFrame({
        'Feature1': np.random.rand(100),
        'Feature2': np.random.rand(100),
        'Class': np.random.choice([2, 4], size=100)
    })

    # Creazione e uso del K-Fold
    kfold = KFold(n_splits=n_splits, random_state=random_state)
    folds = kfold.split(data)

    print("\nSuddivisione completata! Ecco le dimensioni di ciascun fold:")
    for i, (train, test) in enumerate(folds):
        print(f"Fold {i + 1}: Training Set -> {len(train)} righe, Test Set -> {len(test)} righe")
