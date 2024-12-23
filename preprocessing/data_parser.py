from abc import ABC, abstractmethod
import pandas as pd

class DataParser(ABC):
    """
    Questa classe astratta definisce un'interfaccia comune per i parser di dataset.
    """
    @abstractmethod
    def parse(self, file_path: str) -> pd.DataFrame:
        pass


class CsvDataParser(DataParser):
    """
    Questa classe implementa il parser per i file CSV.
    """
    def parse(self, file_path: str) -> pd.DataFrame:
        print(f"Parsing CSV: {file_path}")
        df = pd.read_csv(file_path)
        # Rimuovere i duplicati basati su 'Sample code number' e impostare come indice
        df_cleaned = df.drop_duplicates(subset="Sample code number").set_index("Sample code number")
        return df_cleaned


class ExcelDataParser(DataParser):
    """
    Questa classe implementa il parser per i file Excel.
    """
    def parse(self, file_path: str) -> pd.DataFrame:
        print(f"Parsing Excel: {file_path}")
        df = pd.read_excel(file_path)
        # Rimuovere i duplicati basati su 'Sample code number' e impostare come indice
        df_cleaned = df.drop_duplicates(subset="Sample code number").set_index("Sample code number")
        return df_cleaned


class JsonDataParser(DataParser):
    """
    Questa classe implementa il parser per i file JSON.
    """
    def parse(self, file_path: str) -> pd.DataFrame:
        print(f"Parsing JSON: {file_path}")
        df = pd.read_json(file_path)
        # Rimuovere i duplicati basati su 'Sample code number' e impostare come indice
        df_cleaned = df.drop_duplicates(subset="Sample code number").set_index("Sample code number")
        return df_cleaned


class TxtDataParser(DataParser):
    """
    Questa classe implementa il parser per i file TXT.
    """
    def parse(self, file_path: str) -> pd.DataFrame:
        print(f"Parsing TXT: {file_path}")
        df = pd.read_csv(file_path, delimiter=',')
        # Rimuovere i duplicati basati su 'Sample code number' e impostare come indice
        df_cleaned = df.drop_duplicates(subset="Sample code number").set_index("Sample code number")
        return df_cleaned


class TsvDataParser(DataParser):
    """
    Questa classe implementa il parser per i file TSV.
    """
    def parse(self, file_path: str) -> pd.DataFrame:
        print(f"Parsing TSV: {file_path}")
        df = pd.read_csv(file_path, delimiter='\t')
        # Rimuovere i duplicati basati su 'Sample code number' e impostare come indice
        df_cleaned = df.drop_duplicates(subset="Sample code number").set_index("Sample code number")
        return df_cleaned


class ParserFactory:
    """
    Questa classe fornisce un metodo statico per ottenere il parser corretto in base all'estensione del file.
    """
    @staticmethod
    def get_parser(file_path: str) -> DataParser:
        if file_path.endswith(".csv"):
            return CsvDataParser()
        elif file_path.endswith(".xlsx"):
            return ExcelDataParser()
        elif file_path.endswith(".json"):
            return JsonDataParser()
        elif file_path.endswith(".txt"):
            return TxtDataParser()
        elif file_path.endswith(".tsv"):
            return TsvDataParser()
        else:
            raise RuntimeError("Formato file non supportato")
