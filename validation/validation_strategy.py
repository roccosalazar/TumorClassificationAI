from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class ValidationStrategy(ABC):
    @abstractmethod
    def generate_splits(self, data: pd.DataFrame, labels: pd.Series) -> list[tuple[list[int], list[int]]]:
        """
        Metodo astratto per generare le coppie (predizioni, etichette reali).
        Deve essere implementato nelle sottoclassi.

        Args:
            data (pd.DataFrame): Le feature del dataset.
            labels (pd.Series): Le etichette del dataset.

        Returns:
            np.ndarray: Matrice di predizioni e etichette reali.
        """
        pass
