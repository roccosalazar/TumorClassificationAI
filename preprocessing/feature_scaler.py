import pandas as pd
import numpy as np
from .data_parser import ParserFactory
from .missing_values_handler import MissingValuesStrategyManager

class FeatureScaler:
    """
    Classe per la gestione del feature scaling in un DataFrame.
    """
    @staticmethod
    def normalize(data: pd.DataFrame, exclude_columns: list) -> pd.DataFrame:
        """
        Applica la normalizzazione (range [0, 1]) alle colonne numeriche del dataset, escludendo alcune colonne specifiche.

        Args:
            data (pd.DataFrame): Il dataset da scalare.
            exclude_columns (list): Lista di colonne da escludere dallo scaling.

        Returns:
            pd.DataFrame: Dataset con scaling applicato alle colonne rilevanti.
        """
        columns_to_scale = [col for col in data.columns if col not in exclude_columns]

        data_scaled = data.copy()
        for column in columns_to_scale:
            min_val = data[column].min()
            max_val = data[column].max()
            data_scaled[column] = (data[column] - min_val) / (max_val - min_val)

        return data_scaled

    @staticmethod
    def standardize(data: pd.DataFrame, exclude_columns: list) -> pd.DataFrame:
        """
        Applica la standardizzazione (media 0, deviazione standard 1) alle colonne numeriche del dataset, escludendo alcune colonne specifiche.

        Args:
            data (pd.DataFrame): Il dataset da scalare.
            exclude_columns (list): Lista di colonne da escludere dallo scaling.

        Returns:
            pd.DataFrame: Dataset con scaling applicato alle colonne rilevanti.
        """
        columns_to_scale = [col for col in data.columns if col not in exclude_columns]

        data_scaled = data.copy()
        for column in columns_to_scale:
            mean_val = data[column].mean()
            std_val = data[column].std()
            data_scaled[column] = (data[column] - mean_val) / std_val

        return data_scaled

class FeatureScalerStrategyManager:
    """
    Factory per la gestione dinamica dello scaling delle feature.
    """
    @staticmethod
    def scale_features(strategy: str, data: pd.DataFrame, exclude_columns: list) -> pd.DataFrame:
        """
        Gestisce lo scaling delle feature utilizzando la strategia specificata.

        Args:
            strategy (str): La strategia da applicare ('normalize' o 'standardize').
            data (pd.DataFrame): Il dataset da scalare.
            exclude_columns (list): Lista di colonne da escludere dallo scaling.

        Returns:
            pd.DataFrame: Dataset con scaling applicato.
        """
        if strategy == 'normalize':
            return FeatureScaler.normalize(data, exclude_columns)
        elif strategy == 'standardize':
            return FeatureScaler.standardize(data, exclude_columns)
        else:
            raise ValueError("Strategia non valida. Scegli tra 'normalize' o 'standardize'.")