from typing import List, Tuple, Dict
import numpy as np

class MetricsCalculator:
    def __init__(self):
        """
        Inizializza  la classe MetricsCalculator tramite il costruttore.
        """
        pass

    def calculate_metrics(self, input_data: List[Tuple[List[int], List[int]]]) -> Dict[str, float]:
        """
        Calcola e aggrega le metriche per una lista di coppie (y_real, y_pred).

        Args:
            input_data (List[Tuple[List[int], List[int]]]): Lista di tuple contenenti i valori reali e predetti.

        Returns:
            Dict[str, float]: Dizionario con le metriche aggregate.
        """
        # Inizializza metriche aggregate
        aggregated_metrics = {
            "Accuracy Rate": [],
            "Error Rate": [],
            "Sensitivity": [],
            "Specificity": [],
            "Geometric Mean": [],
            "Area Under Curve": [],
        }

        # Calcola metriche per ogni coppia e aggiungi i valori alle liste aggregate
        for y_real, y_pred in input_data:
            tp, tn, fp, fn = self._confusion_matrix(y_real, y_pred)
            aggregated_metrics["Accuracy Rate"].append(self._accuracy_rate(tp, tn, fp, fn))
            aggregated_metrics["Error Rate"].append(self._error_rate(tp, tn, fp, fn))
            aggregated_metrics["Sensitivity"].append(self._sensitivity(tp, fn))
            aggregated_metrics["Specificity"].append(self._specificity(tn, fp))
            aggregated_metrics["Geometric Mean"].append(self._geometric_mean(tp, tn, fp, fn))
            aggregated_metrics["Area Under Curve"].append(self._area_under_curve(tp, tn, fp, fn))

        # Restituisce le metriche come media delle iterazioni tramite la seguente list comprehension
        return {key: sum(values) / len(values) for key, values in aggregated_metrics.items()}

    def _confusion_matrix(self, y_real: List[int], y_pred: List[int]) -> Tuple[int, int, int, int]:
        """
        Calcola i valori della matrice di confusione.

        Args:
            y_real (List[int]): Valori reali.
            y_pred (List[int]): Valori predetti.

        Returns:
            Tuple[int, int, int, int]: (True Positive, True Negative, False Positive, False Negative).
        """
        tp = sum(1 for r, p in zip(y_real, y_pred) if r == 1 and p == 1)
        tn = sum(1 for r, p in zip(y_real, y_pred) if r == 0 and p == 0)
        fp = sum(1 for r, p in zip(y_real, y_pred) if r == 0 and p == 1)
        fn = sum(1 for r, p in zip(y_real, y_pred) if r == 1 and p == 0)
        return tp, tn, fp, fn

    def _accuracy_rate(self, tp: int, tn: int, fp: int, fn: int) -> float:
        """
        Calcola l'accuracy rate.

        Returns:
            float: Accuracy Rate.
        """
        total = tp + tn + fp + fn
        return (tp + tn) / total if total > 0 else 0.0

    def _error_rate(self, tp: int, tn: int, fp: int, fn: int) -> float:
        """
        Calcola l'error rate.

        Returns:
            float: Error Rate.
        """
        total = tp + tn + fp + fn
        return (fp + fn) / total if total > 0 else 0.0

    def _sensitivity(self, tp: int, fn: int) -> float:
        """
        Calcola la sensitivity (recall o true positive rate).

        Returns:
            float: Sensitivity.
        """
        actual_positive = tp + fn
        return tp / actual_positive if actual_positive > 0 else 0.0

    def _specificity(self, tn: int, fp: int) -> float:
        """
        Calcola la specificity (true negative rate).

        Returns:
            float: Specificity.
        """
        actual_negative = tn + fp
        return tn / actual_negative if actual_negative > 0 else 0.0

    def _geometric_mean(self, tp: int, tn: int, fp: int, fn: int) -> float:
        """
        Calcola la Geometric Mean (G-Mean).

        Returns:
            float: Geometric Mean.
        """
        sensitivity = self._sensitivity(tp, fn)
        specificity = self._specificity(tn, fp)
        return np.sqrt(sensitivity * specificity)

    def _area_under_curve(self, tp: int, tn: int, fp: int, fn: int) -> float:
        """
        Calcola l'Area Under the Curve (AUC) basata su sensitivity e specificity.

        Returns:
            float: Area Under the Curve.
        """
        sensitivity = self._sensitivity(tp, fn)
        specificity = self._specificity(tn, fp)
        return (sensitivity + specificity) / 2
