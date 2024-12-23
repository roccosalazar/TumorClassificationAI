from abc import ABC, abstractmethod
import pandas as pd

"""
Questo file contiene un'implementazione di un'interfaccia per i parser di dataset.
Ogni parser è una classe che implementa il metodo parse, che restituisce un DataFrame pandas.
"""

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
        return pd.read_csv(file_path)


class ExcelDataParser(DataParser):
    """
    Questa classe implementa il parser per i file Excel.
    """
    def parse(self, file_path: str) -> pd.DataFrame:
        print(f"Parsing Excel: {file_path}")
        return pd.read_excel(file_path)


class JsonDataParser(DataParser):
    """
    Questa classe implementa il parser per i file JSON.
    """
    def parse(self, file_path: str) -> pd.DataFrame:
        print(f"Parsing JSON: {file_path}")
        return pd.read_json(file_path)


class TxtDataParser(DataParser):
    """
    Questa classe implementa il parser per i file TXT.
    """
    def parse(self, file_path: str) -> pd.DataFrame:
        print(f"Parsing TXT: {file_path}")
        return pd.read_csv(file_path, delimiter=',')


class TsvDataParser(DataParser):
    """
    Questa classe implementa il parser per i file TSV.
    """
    def parse(self, file_path: str) -> pd.DataFrame:
        print(f"Parsing TSV: {file_path}")
        return pd.read_csv(file_path, delimiter='\t')


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

"""
Da usare in questo modo:
parser = ParserFactory.get_parser("version_1.csv")
df = parser.parse("version_1.csv")
print(df)
"""
