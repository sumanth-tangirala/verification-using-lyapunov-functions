"""
Calibration-based threshold selection for 3-way Lyapunov classification.

Uses grid search to find optimal thresholds (c1, c2, Δ) that minimize:
    loss = w * misclassification% + (1-w) * separatrix%
where separatrix% = overlap% + unknown%
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from .models import DualLyapunovNetwork
from .systems import get_system_config, StateEncoder


@dataclass
class CalibrationResult:
    """Results from calibration procedure."""
    c1: float  # Threshold for V1 (basin 0 / failure)
    c2: float  # Threshold for V2 (basin 1 / success)
    delta: float  # Margin

    # Metrics on calibration set
    coverage: float
    misclassification_rate: float
    misclassification_rate_classified: float
    misclassification_rate_basin0: float  # Basin 0 samples misclassified as Basin 1
    misclassification_rate_basin1: float  # Basin 1 samples misclassified as Basin 0
    overlap_rate: float
    unknown_rate: float
    separatrix_rate: float  # overlap + unknown
    loss: float

    precision_basin0: float
    recall_basin0: float
    precision_basin1: float
    recall_basin1: float
    confusion_matrix: Dict[str, Dict[str, int]]

    def save(self, path: Union[str, Path]) -> None:
        """Save calibration results to JSON."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CalibrationResult":
        """Load calibration results from JSON."""
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


class ThreeWayClassifier:
    """
    3-way classifier using dual Lyapunov functions with calibrated thresholds.

    Classification rule (using labels 0=failure, 1=success):
    - Basin 0 (failure): V1(x) ≤ c1 AND V2(x) ≥ c2 + Δ
    - Basin 1 (success): V2(x) ≤ c2 AND V1(x) ≥ c1 + Δ
    - Unknown: otherwise (includes overlap cases)

    Note: V1 corresponds to basin 0 (failure), V2 corresponds to basin 1 (success)
    """

    def __init__(
        self,
        model: DualLyapunovNetwork,
        c1: float,
        c2: float,
        delta: float = 0.0,
        device: str = "cpu",
    ):
        self.model = model
        self.c1 = c1
        self.c2 = c2
        self.delta = delta
        self.device = device

    @torch.no_grad()
    def compute_scores(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute V1 (basin 0) and V2 (basin 1) scores for input states."""
        self.model.eval()
        x = x.to(self.device)
        V1, V2 = self.model(x)
        return V1.squeeze(-1).cpu(), V2.squeeze(-1).cpu()

    def classify(self, x: Tensor) -> Tensor:
        """
        Classify states into Basin 0, Basin 1, or Unknown.

        Args:
            x: Encoded states (N, input_dim)

        Returns:
            labels: (N,) with values 0 (basin 0/failure), 1 (basin 1/success), or -1 (Unknown)
        """
        s1, s2 = self.compute_scores(x)
        return self._classify_from_scores(s1.numpy(), s2.numpy())

    def _classify_from_scores(self, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
        """Classify from precomputed scores."""
        # Basin 0 (failure): s1 ≤ c1 AND s2 ≥ c2 + Δ
        basin0_mask = (s1 <= self.c1) & (s2 >= self.c2 + self.delta)

        # Basin 1 (success): s2 ≤ c2 AND s1 ≥ c1 + Δ
        basin1_mask = (s2 <= self.c2) & (s1 >= self.c1 + self.delta)

        # Overlap (both conditions true)
        overlap_mask = basin0_mask & basin1_mask

        labels = np.full(len(s1), -1, dtype=np.int64)  # Default: Unknown
        labels[basin0_mask] = 0
        labels[basin1_mask] = 1
        labels[overlap_mask] = -1  # Overlap -> Unknown

        return labels

    def classify_with_details(self, x: Tensor) -> Dict[str, np.ndarray]:
        """Classify with detailed outputs."""
        s1, s2 = self.compute_scores(x)
        s1_np, s2_np = s1.numpy(), s2.numpy()

        basin0_mask = (s1_np <= self.c1) & (s2_np >= self.c2 + self.delta)
        basin1_mask = (s2_np <= self.c2) & (s1_np >= self.c1 + self.delta)
        overlap_mask = basin0_mask & basin1_mask

        labels = np.full(len(s1_np), -1, dtype=np.int64)
        labels[basin0_mask] = 0
        labels[basin1_mask] = 1
        labels[overlap_mask] = -1

        return {
            "labels": labels,
            "s1": s1_np,
            "s2": s2_np,
            "basin0_mask": basin0_mask,
            "basin1_mask": basin1_mask,
            "overlap_mask": overlap_mask,
        }


def detect_delimiter(filepath: Path) -> str:
    """Detect the delimiter used in a text file."""
    with open(filepath, "r") as f:
        first_line = f.readline().strip()

    if "\t" in first_line:
        return "\t"
    elif "," in first_line:
        return ","
    else:
        return None  # Whitespace


def parse_calibration_file(
    filepath: Path,
    state_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse calibration file robustly.

    Args:
        filepath: Path to calibration file
        state_dim: Expected state dimension

    Returns:
        states: (N, state_dim) array
        labels: (N,) array with values in {0, 1}
    """
    filepath = Path(filepath)
    delimiter = detect_delimiter(filepath)

    try:
        if delimiter is None:
            data = np.loadtxt(filepath, dtype=str)
        else:
            data = np.loadtxt(filepath, delimiter=delimiter, dtype=str)
    except Exception as e:
        raise ValueError(f"Failed to parse {filepath}: {e}")

    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_cols = data.shape[1]

    if n_cols < state_dim + 1:
        raise ValueError(
            f"File has {n_cols} columns, expected at least {state_dim + 1}"
        )

    # Extract states (first state_dim columns)
    states = data[:, :state_dim].astype(np.float64)

    # Extract labels (last column)
    raw_labels = data[:, -1]
    labels = _map_labels(raw_labels)

    return states, labels


def _map_labels(raw_labels: np.ndarray) -> np.ndarray:
    """Map raw labels to {0, 1}."""
    labels = np.zeros(len(raw_labels), dtype=np.int64)

    for i, label in enumerate(raw_labels):
        label_str = str(label).strip().lower()

        try:
            label_int = int(float(label_str))
            # Assume 0=failure, 1=success
            labels[i] = label_int
            continue
        except ValueError:
            pass

        # String matching
        if "0" in label_str or "fail" in label_str or "basin0" in label_str:
            labels[i] = 0
        elif "1" in label_str or "success" in label_str or "basin1" in label_str:
            labels[i] = 1
        else:
            labels[i] = 0  # Default

    return labels


def evaluate_thresholds(
    s1: np.ndarray,
    s2: np.ndarray,
    labels: np.ndarray,
    c1: float,
    c2: float,
    delta: float,
) -> Dict[str, float]:
    """
    Evaluate classifier with given thresholds.

    Returns metrics dict.
    """
    n = len(labels)

    # Apply classification rule
    # Basin 0: s1 ≤ c1 AND s2 ≥ c2 + Δ
    basin0_mask = (s1 <= c1) & (s2 >= c2 + delta)
    # Basin 1: s2 ≤ c2 AND s1 ≥ c1 + Δ
    basin1_mask = (s2 <= c2) & (s1 >= c1 + delta)
    # Overlap
    overlap_mask = basin0_mask & basin1_mask

    # Predictions: -1=Unknown, 0=Basin0, 1=Basin1
    preds = np.full(n, -1, dtype=np.int64)
    preds[basin0_mask] = 0
    preds[basin1_mask] = 1
    preds[overlap_mask] = -1  # Overlap -> Unknown

    # Rates
    classified_mask = (preds != -1)
    unknown_mask = (preds == -1) & ~overlap_mask

    coverage = classified_mask.sum() / n
    overlap_rate = overlap_mask.sum() / n
    unknown_rate = (~classified_mask).sum() / n  # includes overlap
    separatrix_rate = overlap_rate + (unknown_mask.sum() / n)  # overlap + pure unknown

    # Misclassification
    if classified_mask.sum() > 0:
        misclass_classified = (preds[classified_mask] != labels[classified_mask]).mean()
    else:
        misclass_classified = 0.0

    misclass_overall = (preds != labels).sum() / n  # Unknown counts as wrong

    # Per-basin precision/recall
    pred_basin0 = (preds == 0)
    true_basin0 = (labels == 0)
    tp0 = (pred_basin0 & true_basin0).sum()
    fp0 = (pred_basin0 & ~true_basin0).sum()
    fn0 = (~pred_basin0 & true_basin0).sum()

    precision_basin0 = tp0 / (tp0 + fp0) if (tp0 + fp0) > 0 else 0.0
    recall_basin0 = tp0 / (tp0 + fn0) if (tp0 + fn0) > 0 else 0.0

    pred_basin1 = (preds == 1)
    true_basin1 = (labels == 1)
    tp1 = (pred_basin1 & true_basin1).sum()
    fp1 = (pred_basin1 & ~true_basin1).sum()
    fn1 = (~pred_basin1 & true_basin1).sum()

    precision_basin1 = tp1 / (tp1 + fp1) if (tp1 + fp1) > 0 else 0.0
    recall_basin1 = tp1 / (tp1 + fn1) if (tp1 + fn1) > 0 else 0.0

    # Per-basin misclassification rates (for balanced loss)
    # Basin 0 misclass = FP0 / total true basin 0 (wrongly classified as basin 1)
    n_true_basin0 = true_basin0.sum()
    n_true_basin1 = true_basin1.sum()
    misclass_basin0 = fp1 / n_true_basin0 if n_true_basin0 > 0 else 0.0  # basin0 samples classified as basin1
    misclass_basin1 = fp0 / n_true_basin1 if n_true_basin1 > 0 else 0.0  # basin1 samples classified as basin0

    # Confusion matrix
    confusion = {
        "true_basin0": {
            "pred_unknown": int(((labels == 0) & (preds == -1)).sum()),
            "pred_basin0": int(((labels == 0) & (preds == 0)).sum()),
            "pred_basin1": int(((labels == 0) & (preds == 1)).sum()),
        },
        "true_basin1": {
            "pred_unknown": int(((labels == 1) & (preds == -1)).sum()),
            "pred_basin0": int(((labels == 1) & (preds == 0)).sum()),
            "pred_basin1": int(((labels == 1) & (preds == 1)).sum()),
        },
    }

    return {
        "coverage": float(coverage),
        "misclassification_rate": float(misclass_overall),
        "misclassification_rate_classified": float(misclass_classified),
        "misclassification_rate_basin0": float(misclass_basin0),
        "misclassification_rate_basin1": float(misclass_basin1),
        "overlap_rate": float(overlap_rate),
        "unknown_rate": float(unknown_rate),
        "separatrix_rate": float(separatrix_rate),
        "precision_basin0": float(precision_basin0),
        "recall_basin0": float(recall_basin0),
        "precision_basin1": float(precision_basin1),
        "recall_basin1": float(recall_basin1),
        "confusion_matrix": confusion,
    }


def compute_loss(metrics: Dict[str, float], w: float = 0.9, balanced: bool = False) -> float:
    """
    Compute calibration loss.

    loss = w * misclassification% + (1-w) * separatrix%
    where separatrix% = overlap% + unknown%

    Args:
        metrics: Dictionary of evaluation metrics
        w: Weight for misclassification in loss (default 0.9)
        balanced: If True, use balanced misclassification rate (sum of per-basin rates)
                  which gives equal weight to both basins regardless of class imbalance

    Returns:
        Calibration loss value
    """
    if balanced:
        # Balanced loss: sum of per-basin misclassification rates
        # This gives equal weight to both basins regardless of class imbalance
        misclass = metrics["misclassification_rate_basin0"] + metrics["misclassification_rate_basin1"]
    else:
        misclass = metrics["misclassification_rate_classified"]
    separatrix = metrics["separatrix_rate"]
    return w * misclass + (1 - w) * separatrix


def grid_search_thresholds(
    s1: np.ndarray,
    s2: np.ndarray,
    labels: np.ndarray,
    n_quantiles: int = 20,
    n_delta: int = 10,
    w: float = 0.9,
    balanced: bool = False,
    verbose: bool = True,
) -> Tuple[float, float, float, Dict[str, float]]:
    """
    Grid search for optimal thresholds (c1, c2, Δ).

    Search ranges:
    - c1: quantiles of s1 (1% to 99%)
    - c2: quantiles of s2 (1% to 99%)
    - Δ: 0 to some scale based on score range

    Selection loss:
        loss = w * misclassification% + (1-w) * separatrix%
    where separatrix% = overlap% + unknown%

    Args:
        s1, s2: Score arrays
        labels: Ground truth labels (0 or 1)
        n_quantiles: Number of quantile points for c1, c2
        n_delta: Number of delta values to try
        w: Weight for misclassification in loss (default 0.9)
        balanced: If True, use balanced misclassification (sum of per-basin rates)
        verbose: Print progress

    Returns:
        best_c1, best_c2, best_delta, best_metrics
    """
    # Define search ranges
    quantiles = np.linspace(0.01, 0.99, n_quantiles)
    c1_range = np.quantile(s1, quantiles)
    c2_range = np.quantile(s2, quantiles)

    # Delta range based on score magnitudes
    score_scale = max(np.std(s1), np.std(s2))
    delta_range = np.linspace(0, score_scale * 2, n_delta)

    best_loss = float("inf")
    best_c1, best_c2, best_delta = 0.0, 0.0, 0.0
    best_metrics = None

    total_iters = len(c1_range) * len(c2_range) * len(delta_range)

    if verbose:
        print(f"Grid search: {len(c1_range)} x {len(c2_range)} x {len(delta_range)} = {total_iters} combinations")

    for c1 in c1_range:
        for c2 in c2_range:
            for delta in delta_range:
                metrics = evaluate_thresholds(s1, s2, labels, c1, c2, delta)
                loss = compute_loss(metrics, w, balanced=balanced)

                if loss < best_loss:
                    best_loss = loss
                    best_c1 = c1
                    best_c2 = c2
                    best_delta = delta
                    best_metrics = metrics

    best_metrics["loss"] = best_loss

    return float(best_c1), float(best_c2), float(best_delta), best_metrics


def calibrate(
    model: DualLyapunovNetwork,
    cal_file: Union[str, Path],
    system_name: str,
    n_quantiles: int = 20,
    n_delta: int = 10,
    w: float = 0.9,
    balanced: bool = False,
    device: str = "cpu",
    verbose: bool = True,
) -> CalibrationResult:
    """
    Full calibration procedure.

    1. Load and parse calibration file
    2. Compute V1, V2 scores
    3. Grid search for optimal (c1, c2, Δ)
    4. Return results

    Args:
        model: Trained DualLyapunovNetwork
        cal_file: Path to calibration file
        system_name: Name of the system
        n_quantiles: Grid resolution for c1, c2
        n_delta: Grid resolution for Δ
        w: Weight for misclassification in loss (default 0.9)
        balanced: If True, use balanced misclassification (sum of per-basin rates)
        device: Device for model inference
        verbose: Print detailed results

    Returns:
        CalibrationResult with thresholds and metrics
    """
    cal_file = Path(cal_file)

    # Get system config and encoder
    config = get_system_config(system_name)
    encoder = StateEncoder(config)

    if verbose:
        print(f"\n{'='*60}")
        print(f"CALIBRATION: {system_name}")
        print(f"{'='*60}")
        print(f"File: {cal_file}")
        print(f"Loss weight (w): {w}")
        print(f"Balanced loss: {balanced}")

    # 1. Load and parse
    states, labels = parse_calibration_file(cal_file, config.state_dim)

    # Filter to valid labels
    valid_mask = (labels == 0) | (labels == 1)
    states = states[valid_mask]
    labels = labels[valid_mask]

    if verbose:
        n_basin0 = (labels == 0).sum()
        n_basin1 = (labels == 1).sum()
        print(f"Loaded {len(labels)} samples: {n_basin0} basin 0, {n_basin1} basin 1")

    # 2. Compute scores
    model.eval()
    states_tensor = torch.from_numpy(states).float()
    states_encoded = encoder.encode(states_tensor)

    with torch.no_grad():
        states_encoded = states_encoded.to(device)
        V1, V2 = model(states_encoded)
        s1 = V1.squeeze(-1).cpu().numpy()
        s2 = V2.squeeze(-1).cpu().numpy()

    if verbose:
        print(f"\nScore statistics:")
        print(f"  V1 (basin 0): min={s1.min():.4f}, max={s1.max():.4f}, mean={s1.mean():.4f}")
        print(f"  V2 (basin 1): min={s2.min():.4f}, max={s2.max():.4f}, mean={s2.mean():.4f}")

    # 3. Grid search
    if verbose:
        print(f"\nRunning grid search...")

    c1, c2, delta, metrics = grid_search_thresholds(
        s1, s2, labels, n_quantiles, n_delta, w, balanced, verbose
    )

    if verbose:
        print(f"\nOptimal thresholds:")
        print(f"  c1 = {c1:.6f}")
        print(f"  c2 = {c2:.6f}")
        print(f"  Δ  = {delta:.6f}")

    # 4. Create result
    result = CalibrationResult(
        c1=c1,
        c2=c2,
        delta=delta,
        **metrics,
    )

    if verbose:
        print_calibration_results(result)

    return result


def print_calibration_results(result: CalibrationResult) -> None:
    """Pretty print calibration results."""
    print(f"\n{'='*60}")
    print("CALIBRATION RESULTS")
    print(f"{'='*60}")

    print(f"\nThresholds:")
    print(f"  c1 (basin 0) = {result.c1:.6f}")
    print(f"  c2 (basin 1) = {result.c2:.6f}")
    print(f"  Δ (margin)   = {result.delta:.6f}")

    print(f"\nPerformance Metrics:")
    print(f"  Coverage: {result.coverage:.4f} ({result.coverage*100:.1f}%)")
    print(f"  Misclassification (overall): {result.misclassification_rate:.4f}")
    print(f"  Misclassification (classified only): {result.misclassification_rate_classified:.4f}")
    print(f"  Overlap rate: {result.overlap_rate:.4f}")
    print(f"  Unknown rate: {result.unknown_rate:.4f}")
    print(f"  Separatrix rate (overlap+unknown): {result.separatrix_rate:.4f}")
    print(f"  Loss: {result.loss:.6f}")

    print(f"\nPer-Basin Metrics:")
    print(f"  Basin 0 - Precision: {result.precision_basin0:.4f}, Recall: {result.recall_basin0:.4f}")
    print(f"  Basin 1 - Precision: {result.precision_basin1:.4f}, Recall: {result.recall_basin1:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"  {'':15} {'Unknown':>10} {'Basin 0':>10} {'Basin 1':>10}")
    cm = result.confusion_matrix
    print(f"  {'True Basin 0':15} {cm['true_basin0']['pred_unknown']:>10} "
          f"{cm['true_basin0']['pred_basin0']:>10} {cm['true_basin0']['pred_basin1']:>10}")
    print(f"  {'True Basin 1':15} {cm['true_basin1']['pred_unknown']:>10} "
          f"{cm['true_basin1']['pred_basin0']:>10} {cm['true_basin1']['pred_basin1']:>10}")

    print(f"{'='*60}\n")


def create_classifier_from_calibration(
    model: DualLyapunovNetwork,
    calibration: CalibrationResult,
    device: str = "cpu",
) -> ThreeWayClassifier:
    """Create a ThreeWayClassifier from calibration results."""
    return ThreeWayClassifier(
        model=model,
        c1=calibration.c1,
        c2=calibration.c2,
        delta=calibration.delta,
        device=device,
    )
