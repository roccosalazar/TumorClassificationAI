import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from .metrics_calculator import MetricsCalculator

class PerformanceMetricsVisualizer:
    def __init__(self, input_data: List[Tuple[List[int], List[int]]]):
        """
        Inizializza l'oggetto per visualizzare le metriche.

        Args:
            input_data (List[Tuple[List[int], List[int]]]): Una lista di tuple (y_real, y_pred).
        """
        self.input_data = input_data
        self.calculator = MetricsCalculator()
        self.metrics = {}

    def visualize_metrics(self) -> None:
        """
        Calcola e visualizza tutte le metriche disponibili.
        """
        # Calcola le metriche utilizzando il MetricsCalculator
        self.metrics = self.calculator.calculate_metrics(self.input_data)

        # Plotta le metriche
        self._plot_metrics(self.metrics)

    def save(self, filename: str = "metrics.xlsx") -> None:
        """
        Salva le metriche calcolate in un file Excel.

        Args:
            filename (str): Nome del file Excel. Default: "metrics.xlsx".
        """
        if not self.metrics:
            print("Nessuna metrica da salvare. Esegui prima 'visualize_metrics()'.")
            return

        # Salva le metriche in un file Excel
        metrics_df = pd.DataFrame(self.metrics.items(), columns=["Metric", "Value"])
        metrics_df.to_excel(filename, index=False, engine="openpyxl")
        print(f"Metriche salvate in {filename}")

        """filename: Nome del file in cui salvare i dati,
        Non include l'indice del DataFrame nel file Excel,
        Specifica che viene utilizzato il motore openpyxl per scrivere il file"""

    def _plot_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Plotta le metriche in un grafico a barre.

        Args:
            metrics (dict): Dizionario contenente i valori delle metriche.
        """
        plt.figure(figsize=(10, 6))
        plt.bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'orange', 'red', 'purple', 'cyan'])
        plt.title("Performance Metrics")
        plt.ylabel("Value")
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()

if __name__ == "__main__":
    # Dati di esempio
    validation = [
    ([1, 0, 1, 1, 0, 1], [1, 0, 1, 1, 0, 1]),
    ([0, 0, 1, 0, 0, 1], [0, 1, 1, 0, 0, 0]),
    ([1, 1, 0, 1, 0, 1], [1, 1, 0, 0, 0, 1]),
    ([0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0]),
    ([1, 0, 0, 1, 1, 1], [1, 0, 0, 1, 1, 0]),
    ]

    # Creazione dell'oggetto PerformanceMetricsVisualizer
    visualizer = PerformanceMetricsVisualizer(validation)

    # Calcolo e visualizzazione delle metriche
    visualizer.visualize_metrics()

    # Salvataggio delle metriche in un file Excel
    visualizer.save("metrics_output.xlsx")

