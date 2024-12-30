from abc import ABC, abstractmethod
import pandas as pd
import numpy as np



# Classe astratta Validation
class Validation(ABC):
    @abstractmethod
    def split(self, data: pd.DataFrame):
        """
        Metodo astratto per suddividere il dataset in training e test set.
        """
        pass


# Classe Holdout che implementa Validation
class Holdout(Validation):
    def __init__(self, test_size, random_state=None):
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
        Divide il dataset in training e test set.

        Args:
            data (pd.DataFrame): Dataset completo.

        Returns:
            tuple: Training set e test set.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        shuffled_indices = np.random.permutation(len(data))  # Mescola gli indici

        test_set_size = int(len(data) * self.test_size)  # Dimensione del test set
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]

        train = data.iloc[train_indices]
        test = data.iloc[test_indices]

        return train, test


if __name__ == "__main__":
    # Interfaccia per configurare il metodo Holdout
    print("Configura la validazione Holdout:")

    # Input per il test_size
    while True:
        try:
            test_size = input("Inserisci la proporzione per il test set (tra 0.1 e 0.5, default 0.2): ")
            if not test_size:  # Se l'utente non inserisce nulla
                test_size = 0.2  # Assegna il valore di default
            else:
                test_size = float(test_size)
                if not (0.1 <= test_size <= 0.5):
                    raise ValueError("La proporzione deve essere tra 0.1 e 0.5.")
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

    print(f"Configurazione scelta: test_size={test_size}, random_state={random_state}")

    # Dataset simulato
    data = pd.DataFrame({
        'Feature1': np.random.rand(100),
        'Feature2': np.random.rand(100),
        'Class': np.random.choice([2, 4], size=100)
    })

    # Creazione e uso dell'Holdout
    holdout = Holdout(test_size=test_size, random_state=random_state)
    train, test = holdout.split(data)

    print(f"\nDataset diviso in training e test set:")
    print(f"Training Set: {len(train)} righe")
    print(f"Test Set: {len(test)} righe")
