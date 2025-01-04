import numpy as np


class MetricsCalculator:
    def __init__(self, input_data, method="holdout"):
        """
        Inizializza l'input e il metodo di validazione.

        Args:
            input_data: Una matrice contenente gli array numpy di y_real e y_pred.
            method (str): Metodo di validazione ("holdout" o "Leave-p-out Cross Validation").
        """
        self.input_data = input_data
        self.method = method

    def accuracy(self, y_real, y_pred):
        """Calcola il tasso di accuratezza."""
        return np.mean(y_real == y_pred)

    def error_rate(self, y_real, y_pred):
        """Calcola il tasso di errore."""
        return 1 - self.accuracy(y_real, y_pred)

    def sensitivity(self, y_real, y_pred):
        """Calcola la sensibilità (recall per la classe positiva)."""
        TP = np.sum((y_real == 1) & (y_pred == 1))
        FN = np.sum((y_real == 1) & (y_pred == 0))
        return TP / (TP + FN) if (TP + FN) > 0 else 0

    def _calculate_metrics(self, y_real, y_pred):
        """Calcola le metriche per una singola coppia di array."""
        return {
            "Accuracy": self.accuracy(y_real, y_pred),
            "Error Rate": self.error_rate(y_real, y_pred),
            "Sensitivity": self.sensitivity(y_real, y_pred)
        }

    def calculate_metrics(self):
        """
        Calcola le metriche in base al metodo di validazione.
        """
        if self.method == "holdout":
            y_real, y_pred = self.input_data[0]
            return self._calculate_metrics(y_real, y_pred)

        elif self.method == "Leave-p-out Cross Validation":
            metrics_list = []
            total_tp = 0
            total_fn = 0

            for y_real, y_pred in self.input_data:
                metrics = self._calculate_metrics(y_real, y_pred)
                metrics_list.append(metrics)
                total_tp += np.sum((y_real == 1) & (y_pred == 1))
                total_fn += np.sum((y_real == 1) & (y_pred == 0))

            aggregated_metrics = {
                metric: np.mean([m[metric] for m in metrics_list])
                for metric in metrics_list[0]
            }
            aggregated_metrics["Sensitivity"] = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

            return aggregated_metrics

        else:
            raise ValueError("Metodo non supportato. Usa 'holdout' o 'Leave-p-out Cross Validation'.")

