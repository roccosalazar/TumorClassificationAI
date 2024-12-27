from abc import ABC, abstractmethod
import pandas as pd

class Validation(ABC):
    @abstractmethod
    def split(self, data: pd.DataFrame):
        """
        Metodo astratto per suddividere il dataset in training e test set.
        """
        pass

    @abstractmethod
    def evaluate(self, model, data: pd.DataFrame, target_column: str):
        """
        Metodo astratto per valutare un modello.
        """
        pass