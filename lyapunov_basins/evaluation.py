"""Evaluation metrics for dual Lyapunov networks."""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from .models import DualLyapunovNetwork


class LyapunovEvaluator:
    """
    Evaluator for trained dual Lyapunov networks.

    Computes:
    1. Classification metrics: confusion matrix, accuracy, precision, recall,
       specificity, F1 (after removing separatrix samples)
    2. Lyapunov violation metrics: how often V increases along trajectories

    Uses calibrated thresholds (c1, c2, delta) for 3-way classification:
    - Basin 0: V1(x) <= c1 AND V2(x) >= c2 + delta
    - Basin 1: V2(x) <= c2 AND V1(x) >= c1 + delta
    - Separatrix: otherwise (includes overlap and unknown)
    """

    def __init__(
        self,
        model: DualLyapunovNetwork,
        c1: float,
        c2: float,
        delta: float = 0.0,
        device: str = "cuda",
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained DualLyapunovNetwork
            c1: Calibrated threshold for V1 (basin 0)
            c2: Calibrated threshold for V2 (basin 1)
            delta: Calibrated margin
            device: Device for computation
        """
        self.model = model
        self.c1 = c1
        self.c2 = c2
        self.delta = delta
        self.device = device

    def _classify_from_scores(self, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
        """
        Classify from precomputed scores using calibrated thresholds.

        Args:
            s1: V1 scores (N,)
            s2: V2 scores (N,)

        Returns:
            labels: (N,) with values 0 (basin 0), 1 (basin 1), or -1 (separatrix)
        """
        # Basin 0: s1 <= c1 AND s2 >= c2 + delta
        basin0_mask = (s1 <= self.c1) & (s2 >= self.c2 + self.delta)

        # Basin 1: s2 <= c2 AND s1 >= c1 + delta
        basin1_mask = (s2 <= self.c2) & (s1 >= self.c1 + self.delta)

        # Overlap (both conditions true)
        overlap_mask = basin0_mask & basin1_mask

        labels = np.full(len(s1), -1, dtype=np.int64)  # Default: separatrix
        labels[basin0_mask] = 0
        labels[basin1_mask] = 1
        labels[overlap_mask] = -1  # Overlap -> separatrix

        return labels

    @torch.no_grad()
    def evaluate_classification(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate classification on states using calibrated thresholds.

        Uses 3-way classification:
        - Basin 0: V1(x) <= c1 AND V2(x) >= c2 + delta
        - Basin 1: V2(x) <= c2 AND V1(x) >= c1 + delta
        - Separatrix: otherwise

        Separatrix samples are removed before computing classification metrics.

        Args:
            dataloader: DataLoader yielding (x_enc, label) or (x_t_enc, x_tp1_enc, label)

        Returns:
            Dictionary with separatrix_pct, confusion matrix, accuracy, precision,
            recall, specificity, F1
        """
        self.model.eval()

        all_s1 = []
        all_s2 = []
        all_labels = []

        for batch in dataloader:
            # Handle both 2-tuple (state dataset) and 3-tuple (trajectory dataset)
            if len(batch) == 2:
                x_enc, labels = batch
            else:
                x_enc, _, labels = batch  # Use x_t for classification

            x_enc = x_enc.to(self.device)

            V1, V2 = self.model(x_enc)
            all_s1.append(V1.squeeze(-1).cpu().numpy())
            all_s2.append(V2.squeeze(-1).cpu().numpy())
            all_labels.append(labels.numpy())

        s1 = np.concatenate(all_s1)
        s2 = np.concatenate(all_s2)
        labels = np.concatenate(all_labels)

        # 3-way classification
        preds = self._classify_from_scores(s1, s2)

        # Identify separatrix samples
        separatrix_mask = (preds == -1)
        classified_mask = ~separatrix_mask

        total_samples = len(labels)
        separatrix_count = separatrix_mask.sum()
        classified_count = classified_mask.sum()

        separatrix_pct = separatrix_count / total_samples if total_samples > 0 else 0.0

        # Filter to classified samples only
        preds_classified = preds[classified_mask]
        labels_classified = labels[classified_mask]

        # Compute metrics on classified samples
        # Positive = success (basin 1), Negative = failure (basin 0)
        if classified_count > 0:
            tp = ((preds_classified == 1) & (labels_classified == 1)).sum()
            tn = ((preds_classified == 0) & (labels_classified == 0)).sum()
            fp = ((preds_classified == 1) & (labels_classified == 0)).sum()
            fn = ((preds_classified == 0) & (labels_classified == 1)).sum()

            accuracy = (tp + tn) / classified_count
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            confusion_matrix = {
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
            }
        else:
            accuracy = 0.0
            precision = recall = specificity = f1 = 0.0
            confusion_matrix = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

        return {
            "total_samples": int(total_samples),
            "separatrix_count": int(separatrix_count),
            "classified_count": int(classified_count),
            "separatrix_pct": float(separatrix_pct),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1": float(f1),
            "confusion_matrix": confusion_matrix,
        }

    @torch.no_grad()
    def evaluate_lyapunov_violations(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate Lyapunov decrease condition violations.

        For basin i trajectories, counts where V_i(x_{t+1}) > V_i(x_t).

        Args:
            dataloader: DataLoader yielding (x_t_enc, x_tp1_enc, label)

        Returns:
            Dictionary with violation rates and mean violation magnitudes
        """
        self.model.eval()

        violations_0 = []
        violations_1 = []
        total_0 = 0
        total_1 = 0

        for batch in dataloader:
            x_t_enc, x_tp1_enc, labels = batch
            x_t_enc = x_t_enc.to(self.device)
            x_tp1_enc = x_tp1_enc.to(self.device)

            V_0_t, V_1_t = self.model(x_t_enc)
            V_0_tp1, V_1_tp1 = self.model(x_tp1_enc)

            # Compute violations (V increased)
            delta_V_0 = (V_0_tp1 - V_0_t).squeeze(-1).cpu()
            delta_V_1 = (V_1_tp1 - V_1_t).squeeze(-1).cpu()

            mask_0 = (labels == 0)
            mask_1 = (labels == 1)

            # Positive delta = violation
            if mask_0.sum() > 0:
                violations_0.append(F.relu(delta_V_0[mask_0]))
                total_0 += mask_0.sum().item()

            if mask_1.sum() > 0:
                violations_1.append(F.relu(delta_V_1[mask_1]))
                total_1 += mask_1.sum().item()

        # Concatenate violations
        if violations_0:
            violations_0 = torch.cat(violations_0)
            violation_rate_0 = (violations_0 > 0).float().mean().item()
            mean_violation_0 = violations_0.mean().item()
        else:
            violation_rate_0 = 0.0
            mean_violation_0 = 0.0

        if violations_1:
            violations_1 = torch.cat(violations_1)
            violation_rate_1 = (violations_1 > 0).float().mean().item()
            mean_violation_1 = violations_1.mean().item()
        else:
            violation_rate_1 = 0.0
            mean_violation_1 = 0.0

        return {
            "violation_rate_0": violation_rate_0,
            "violation_rate_1": violation_rate_1,
            "mean_violation_0": mean_violation_0,
            "mean_violation_1": mean_violation_1,
            "total_pairs_0": total_0,
            "total_pairs_1": total_1,
        }

    @torch.no_grad()
    def evaluate_all(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Run all evaluations.

        Args:
            dataloader: DataLoader yielding (x_t_enc, x_tp1_enc, label)

        Returns:
            Combined dictionary with all metrics
        """
        classification_metrics = self.evaluate_classification(dataloader)
        violation_metrics = self.evaluate_lyapunov_violations(dataloader)

        return {**classification_metrics, **violation_metrics}


def print_evaluation_results(metrics: Dict[str, float]) -> None:
    """Pretty print evaluation metrics."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    if "separatrix_pct" in metrics:
        print("\nSample Distribution:")
        print(f"  Total samples: {metrics['total_samples']}")
        print(f"  Separatrix samples: {metrics['separatrix_count']} ({metrics['separatrix_pct']*100:.2f}%)")
        print(f"  Classified samples: {metrics['classified_count']}")

    if "accuracy" in metrics:
        print("\nClassification Metrics (on classified samples):")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  F1:          {metrics['f1']:.4f}")

    if "confusion_matrix" in metrics:
        cm = metrics["confusion_matrix"]
        print("\n  Confusion Matrix (positive=success, negative=failure):")
        print(f"    {'':15} {'Pred Fail':>12} {'Pred Success':>12}")
        print(f"    {'Actual Fail':15} {cm['tn']:>12} {cm['fp']:>12}")
        print(f"    {'Actual Success':15} {cm['fn']:>12} {cm['tp']:>12}")

    if "violation_rate_0" in metrics:
        print("\nLyapunov Violation Metrics:")
        print(f"  Basin 0 - Violation Rate: {metrics['violation_rate_0']:.4f}, "
              f"Mean Violation: {metrics['mean_violation_0']:.6f}")
        print(f"  Basin 1 - Violation Rate: {metrics['violation_rate_1']:.4f}, "
              f"Mean Violation: {metrics['mean_violation_1']:.6f}")

    print("=" * 60 + "\n")
