import pandas as pd
from preprocessing.data_parser import ParserFactory

class MissingValuesHandler:
    """
    Classe per la gestione dei valori mancanti in un DataFrame.
    """
    @staticmethod
    def convert_numeric_columns(data: pd.DataFrame) -> pd.DataFrame:
        """
        Converte le colonne numeriche rappresentate come stringhe in numeri float.

        Args:
            data (pd.DataFrame): Il dataset con possibili valori numerici rappresentati come stringhe.

        Returns:
            pd.DataFrame: Dataset con colonne numeriche convertite in float.
        """
        for column in data.columns:
            try:
                data[column] = pd.to_numeric(data[column], errors='coerce')
            except ValueError:
                pass
        return data

    @staticmethod
    def remove_missing_rows(data: pd.DataFrame) -> pd.DataFrame:
        """
        Rimuove le righe con valori mancanti.

        Args:
            data (pd.DataFrame): Il dataset con possibili valori mancanti.

        Returns:
            pd.DataFrame: Dataset senza righe con valori mancanti.
        """
        return data.dropna()

    @staticmethod
    def fill_missing_with_mean(data: pd.DataFrame) -> pd.DataFrame:
        """
        Riempie i valori mancanti con la media delle colonne.

        Args:
            data (pd.DataFrame): Il dataset con possibili valori mancanti.

        Returns:
            pd.DataFrame: Dataset con valori mancanti riempiti con la media.
        """
        return data.fillna(data.mean())

    @staticmethod
    def fill_missing_with_median(data: pd.DataFrame) -> pd.DataFrame:
        """
        Riempie i valori mancanti con la mediana delle colonne.

        Args:
            data (pd.DataFrame): Il dataset con possibili valori mancanti.

        Returns:
            pd.DataFrame: Dataset con valori mancanti riempiti con la mediana.
        """
        return data.fillna(data.median())

    @staticmethod
    def fill_missing_with_mode(data: pd.DataFrame) -> pd.DataFrame:
        """
        Riempie i valori mancanti con la moda delle colonne.

        Args:
            data (pd.DataFrame): Il dataset con possibili valori mancanti.

        Returns:
            pd.DataFrame: Dataset con valori mancanti riempiti con la moda.
        """
        mode = data.mode().iloc[0]
        return data.fillna(mode)

class MissingValuesFactory:
    """
    Factory per la gestione dinamica dei valori mancanti.
    """
    @staticmethod
    def handle_missing_values(strategy: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Gestisce i valori mancanti utilizzando la strategia specificata.

        Args:
            strategy (str): La strategia da applicare ('remove', 'mean', 'median', 'mode').
            data (pd.DataFrame): Il dataset con possibili valori mancanti.

        Returns:
            pd.DataFrame: Dataset con valori mancanti gestiti.
        """
        data = MissingValuesHandler.convert_numeric_columns(data)

        if strategy == 'remove':
            return MissingValuesHandler.remove_missing_rows(data)
        elif strategy == 'mean':
            return MissingValuesHandler.fill_missing_with_mean(data)
        elif strategy == 'median':
            return MissingValuesHandler.fill_missing_with_median(data)
        elif strategy == 'mode':
            return MissingValuesHandler.fill_missing_with_mode(data)
        else:
            raise ValueError("Strategia non valida. Scegli tra 'remove', 'mean', 'median', 'mode'.")

if __name__ == "__main__":
    # Esempio di utilizzo della classe MissingValuesFactory
    file_path = "data/version_1.csv"
    parser = ParserFactory.get_parser(file_path)
    data = parser.parse(file_path)

    print("Original Data:")
    print(data.head())

    # Applicare diverse strategie usando la factory
    strategies = ['remove', 'mean', 'median', 'mode']

    for strategy in strategies:
        print(f"\nStrategy: {strategy}")
        processed_data = MissingValuesFactory.handle_missing_values(strategy, data)
        print(processed_data.head())
