#!/usr/bin/env python3
"""
Re-calibrate a trained Lyapunov model with different parameters.

Loads an existing trained model and runs calibration with new settings,
saving results to a parameter-named subdirectory.

Usage:
    python recalibrate.py --run_dir outputs/pendulum/run_20260130_035506 --cal_weight 0.5
    python recalibrate.py --run_dir outputs/pendulum/latest --cal_weight 0.3 --n_quantiles 30
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lyapunov_basins.systems import get_system_config, get_data_dir
from lyapunov_basins.models import DualLyapunovNetwork
from lyapunov_basins.dataset import LyapunovTestDataset
from lyapunov_basins.evaluation import LyapunovEvaluator, print_evaluation_results
from lyapunov_basins.calibration import (
    calibrate,
    CalibrationResult,
    evaluate_thresholds,
    compute_loss,
    parse_calibration_file,
    print_calibration_results,
)
from lyapunov_basins.systems import StateEncoder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-calibrate a trained Lyapunov model with different parameters"
    )

    # Model location
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to the training run directory (e.g., outputs/pendulum/run_XXX or outputs/pendulum/latest)",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default="best_model.pt",
        help="Model file name within run_dir (default: best_model.pt)",
    )

    # Calibration parameters
    parser.add_argument(
        "--cal_weight",
        type=float,
        default=0.5,
        help="Weight for misclassification in calibration loss (default 0.5)",
    )
    parser.add_argument(
        "--n_quantiles",
        type=int,
        default=20,
        help="Number of quantile points for c1, c2 grid search (default 20)",
    )
    parser.add_argument(
        "--n_delta",
        type=int,
        default=10,
        help="Number of delta values to try (default 10)",
    )
    parser.add_argument(
        "--balanced_loss",
        action="store_true",
        help="Use balanced misclassification (sum of per-basin rates) instead of overall rate",
    )
    parser.add_argument(
        "--cal_file",
        type=str,
        default=None,
        help="Calibration file (default: cal_set.txt in data dir)",
    )

    # Manual threshold override
    parser.add_argument(
        "--c1",
        type=float,
        default=None,
        help="Manual c1 threshold (skip grid search if all three are specified)",
    )
    parser.add_argument(
        "--c2",
        type=float,
        default=None,
        help="Manual c2 threshold (skip grid search if all three are specified)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        help="Manual delta margin (skip grid search if all three are specified)",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )

    # Evaluation
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation on test set (only run calibration)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for evaluation (default 512)",
    )

    return parser.parse_args()


def get_cal_dir_name(
    cal_weight: float,
    n_quantiles: int,
    n_delta: int,
    balanced: bool = False,
    c1: float = None,
    c2: float = None,
    delta: float = None,
) -> str:
    """Generate directory name from calibration parameters."""
    if c1 is not None and c2 is not None and delta is not None:
        return f"cal_manual_c1{c1}_c2{c2}_d{delta}"
    bal_suffix = "_bal" if balanced else ""
    return f"cal_w{cal_weight}_nq{n_quantiles}_nd{n_delta}{bal_suffix}"


def load_model(run_dir: Path, model_file: str, device: str):
    """Load trained model and config from run directory."""
    model_path = run_dir / model_file
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Get model config
    config = checkpoint.get("config", {})
    system_name = config.get("system")
    input_dim = config.get("input_dim")
    hidden_sizes = config.get("hidden_sizes")

    # Try to load from config.json if not in checkpoint
    if not all([system_name, input_dim, hidden_sizes]):
        config_path = run_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                full_config = json.load(f)
            system_name = full_config.get("system", system_name)
            model_config = full_config.get("model", {})
            input_dim = model_config.get("input_dim", input_dim)
            hidden_sizes = model_config.get("hidden_sizes", hidden_sizes)

    if not all([system_name, input_dim, hidden_sizes]):
        raise ValueError(
            "Could not determine model configuration. "
            "Ensure config.json exists in run directory."
        )

    # Load attractor points (required for AttractorPosDef model)
    attractor_points_0 = checkpoint.get("attractor_points_0")
    attractor_points_1 = checkpoint.get("attractor_points_1")
    if attractor_points_0 is None or attractor_points_1 is None:
        raise ValueError(
            "Checkpoint does not contain attractor points. "
            "This checkpoint was likely saved with the old PosDef model. "
            "Please retrain with the AttractorPosDef model."
        )

    # Create and load model
    model = DualLyapunovNetwork(
        input_dim=input_dim,
        attractor_points_0=attractor_points_0,
        attractor_points_1=attractor_points_1,
        hidden_sizes=hidden_sizes,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, system_name


def main():
    args = parse_args()

    # Resolve run directory (handle symlinks like 'latest')
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    print(f"\n{'='*60}")
    print("Re-calibration")
    print(f"{'='*60}")
    print(f"Run directory: {run_dir}")

    # Load model
    print(f"\nLoading model from {args.model_file}...")
    model, system_name = load_model(run_dir, args.model_file, args.device)
    print(f"System: {system_name}")
    print(f"Device: {args.device}")

    # Check if manual thresholds are provided
    use_manual = all(x is not None for x in [args.c1, args.c2, args.delta])

    # Create output directory with parameter-based name
    cal_dir_name = get_cal_dir_name(
        args.cal_weight, args.n_quantiles, args.n_delta,
        balanced=args.balanced_loss,
        c1=args.c1, c2=args.c2, delta=args.delta
    )
    output_dir = run_dir / cal_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Determine calibration file
    if args.cal_file:
        cal_file = Path(args.cal_file)
    else:
        data_dir = get_data_dir(system_name)
        cal_file = data_dir / "cal_set.txt"

    if not cal_file.exists():
        raise FileNotFoundError(f"Calibration file not found: {cal_file}")

    # Run calibration
    print(f"\n{'='*60}")
    print("Running Calibration")
    print(f"{'='*60}")

    if use_manual:
        # Manual threshold evaluation
        print(f"Using manual thresholds: c1={args.c1}, c2={args.c2}, delta={args.delta}")

        # Load and encode calibration data
        system_config = get_system_config(system_name)
        encoder = StateEncoder(system_config)
        states, labels = parse_calibration_file(cal_file, system_config.state_dim)

        # Filter valid labels
        import numpy as np
        valid_mask = (labels == 0) | (labels == 1)
        states = states[valid_mask]
        labels = labels[valid_mask]

        n_basin0 = (labels == 0).sum()
        n_basin1 = (labels == 1).sum()
        print(f"Loaded {len(labels)} samples: {n_basin0} basin 0, {n_basin1} basin 1")

        # Compute scores
        import torch
        states_tensor = torch.from_numpy(states).float()
        states_encoded = encoder.encode(states_tensor)

        with torch.no_grad():
            states_encoded = states_encoded.to(args.device)
            V1, V2 = model(states_encoded)
            s1 = V1.squeeze(-1).cpu().numpy()
            s2 = V2.squeeze(-1).cpu().numpy()

        print(f"\nScore statistics:")
        print(f"  V1 (basin 0): min={s1.min():.4f}, max={s1.max():.4f}, mean={s1.mean():.4f}")
        print(f"  V2 (basin 1): min={s2.min():.4f}, max={s2.max():.4f}, mean={s2.mean():.4f}")

        # Evaluate with manual thresholds
        metrics = evaluate_thresholds(s1, s2, labels, args.c1, args.c2, args.delta)
        metrics["loss"] = compute_loss(metrics, w=args.cal_weight, balanced=args.balanced_loss)

        cal_result = CalibrationResult(
            c1=args.c1,
            c2=args.c2,
            delta=args.delta,
            **metrics,
        )
        print_calibration_results(cal_result)
    else:
        # Grid search calibration
        print(f"Parameters: cal_weight={args.cal_weight}, n_quantiles={args.n_quantiles}, n_delta={args.n_delta}, balanced={args.balanced_loss}")

        cal_result = calibrate(
            model=model,
            cal_file=cal_file,
            system_name=system_name,
            n_quantiles=args.n_quantiles,
            n_delta=args.n_delta,
            w=args.cal_weight,
            balanced=args.balanced_loss,
            device=args.device,
            verbose=True,
        )

    # Save calibration results
    cal_path = output_dir / "calibration.json"
    cal_result.save(cal_path)
    print(f"Saved calibration: {cal_path}")

    # Save parameters used
    params_path = output_dir / "params.json"
    params = {
        "cal_weight": args.cal_weight,
        "n_quantiles": args.n_quantiles,
        "n_delta": args.n_delta,
        "balanced_loss": args.balanced_loss,
        "cal_file": str(cal_file),
        "run_dir": str(run_dir),
        "model_file": args.model_file,
        "manual_thresholds": use_manual,
        "c1": args.c1,
        "c2": args.c2,
        "delta": args.delta,
    }
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved parameters: {params_path}")

    # Evaluation on test set
    if not args.skip_eval:
        print(f"\n{'='*60}")
        print("Evaluation on Test Set")
        print(f"{'='*60}")

        # Load test dataset
        test_dataset = LyapunovTestDataset(system_name=system_name)
        num_workers = 4 if args.device == "cuda" else 0
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(args.device == "cuda"),
        )

        # Run evaluation
        evaluator = LyapunovEvaluator(
            model,
            c1=cal_result.c1,
            c2=cal_result.c2,
            delta=cal_result.delta,
            device=args.device,
        )
        test_metrics = evaluator.evaluate_all(test_loader)
        print_evaluation_results(test_metrics)

        # Save metrics
        metrics_path = output_dir / "test_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        print(f"Saved metrics: {metrics_path}")

    print(f"\n{'='*60}")
    print("Re-calibration Complete")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
