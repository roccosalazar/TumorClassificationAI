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
    
    def specificity(self, y_real, y_pred):
        """Calcola la specificità."""
        TN = np.sum((y_real == 0) & (y_pred == 0))
        FP = np.sum((y_real == 0) & (y_pred == 1))
        return TN / (TN + FP) if (TN + FP) > 0 else 0
    
    def geometric_mean(self, y_real, y_pred):
        """Calcola la Geometric Mean (G-Mean)."""
        sensitivity = self.sensitivity(y_real, y_pred)
        specificity = self.specificity(y_real, y_pred)
        return np.sqrt(sensitivity * specificity)
    
    def auc(self, y_real, y_pred):
        """Calcola l'Area Under the Curve (AUC) utilizzando il metodo del trapezio."""
        sorted_indices = np.argsort(y_pred)
        y_real = np.array(y_real)[sorted_indices]
        y_pred = np.array(y_pred)[sorted_indices]

        tpr = []
        fpr = []

        P = np.sum(y_real == 1)
        N = np.sum(y_real == 0)

        for threshold in np.unique(y_pred):
            tp = np.sum((y_pred >= threshold) & (y_real == 1))
            fp = np.sum((y_pred >= threshold) & (y_real == 0))

            tpr.append(tp / P if P > 0 else 0)
            fpr.append(fp / N if N > 0 else 0)

        tpr = np.array(tpr)
        fpr = np.array(fpr)

        return np.trapz(tpr, fpr)


    def _calculate_metrics(self, y_real, y_pred):
        """Calcola le metriche per una singola coppia di array."""
        return {
            "Accuracy": self.accuracy(y_real, y_pred),
            "Error Rate": self.error_rate(y_real, y_pred),
            "Sensitivity": self.sensitivity(y_real, y_pred),
            "Specificity": self.specificity(y_real, y_pred),
            "Geometric Mean": self.geometric_mean(y_real, y_pred),
            "AUC": self.auc(y_real, y_pred)

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
            total_tn = 0
            total_fp = 0

            for y_real, y_pred in self.input_data:
                metrics = self._calculate_metrics(y_real, y_pred)
                metrics_list.append(metrics)
                total_tp += np.sum((y_real == 1) & (y_pred == 1))
                total_fn += np.sum((y_real == 1) & (y_pred == 0))
                total_tn += np.sum((y_real == 0) & (y_pred == 0))
                total_fp += np.sum((y_real == 0) & (y_pred == 1))



            aggregated_metrics = {
                metric: np.mean([m[metric] for m in metrics_list])
                for metric in metrics_list[0]
            }
            aggregated_metrics["Sensitivity"] = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            aggregated_metrics["Specificity"] = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0
            aggregated_metrics["Geometric Mean"] = np.sqrt(aggregated_metrics["Sensitivity"] * aggregated_metrics["Specificity"])



            return aggregated_metrics

        else:
            raise ValueError("Metodo non supportato. Usa 'holdout' o 'Leave-p-out Cross Validation'.")

