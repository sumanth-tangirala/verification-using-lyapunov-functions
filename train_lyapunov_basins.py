#!/usr/bin/env python3
"""
Train dual neural Lyapunov functions for two-basin classification.

Usage:
    python train_lyapunov_basins.py --system pendulum --num_train 1000 --epochs 100
    python train_lyapunov_basins.py --system cartpole --num_train 5000 --margin 2.0
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import shutil
from typing import Dict, Optional

from dotenv import load_dotenv
import torch

# Load environment variables from .env file
load_dotenv()
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from lyapunov_basins.systems import get_system_config, get_data_dir, SYSTEM_CONFIGS
from lyapunov_basins.models import DualLyapunovNetwork, DEFAULT_HIDDEN_SIZES
from lyapunov_basins.dataset import LyapunovTrajectoryDataset, LyapunovTestDataset
from lyapunov_basins.loss import LyapunovContrastiveLoss
from lyapunov_basins.evaluation import LyapunovEvaluator, print_evaluation_results
from lyapunov_basins.calibration import calibrate


def get_cal_dir_name(cal_weight: float, n_quantiles: int = 20, n_delta: int = 10) -> str:
    """Generate directory name from calibration parameters."""
    return f"cal_w{cal_weight}_nq{n_quantiles}_nd{n_delta}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train dual Lyapunov functions for two-basin classification"
    )

    # System selection
    parser.add_argument(
        "--system",
        type=str,
        required=True,
        choices=list(SYSTEM_CONFIGS.keys()),
        help="System to train on",
    )

    # Data
    parser.add_argument(
        "--num_train",
        type=int,
        default=None,
        help="Number of training trajectories (None = all)",
    )
    parser.add_argument(
        "--indices_file",
        type=str,
        default="shuffled_indices_0.txt",
        help="Training indices file in train_test_splits/",
    )
    parser.add_argument(
        "--labels_file",
        type=str,
        default="shuffled_labels_0.txt",
        help="Training labels file in train_test_splits/",
    )

    # Model architecture
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=None,
        help="Hidden layer sizes (default: system-specific)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.01,
        help="Epsilon for PosDef layer",
    )

    # Loss parameters (defaults are system-specific, set to None to use system defaults)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="Margin for Lyapunov decrease condition",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=None,
        help="Margin for contrastive loss (default: system-specific)",
    )
    parser.add_argument(
        "--lambda_decrease",
        type=float,
        default=1.0,
        help="Weight for decrease loss",
    )
    parser.add_argument(
        "--lambda_contrastive",
        type=float,
        default=None,
        help="Weight for contrastive loss (default: system-specific)",
    )
    parser.add_argument(
        "--lambda_attractor",
        type=float,
        default=0.0,
        help="Weight for attractor loss (0 = disabled)",
    )

    # Training (defaults are system-specific, set to None to use system defaults)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: system-specific)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for training (default: system-specific)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: system-specific)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="Weight decay for AdamW (default: system-specific)",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping (0 = disabled)",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("OUTPUT_DIR", "outputs"),
        help="Directory for saving models and logs",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=50,
        help="Save checkpoint every N epochs",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for training",
    )

    # Calibration
    parser.add_argument(
        "--cal_file",
        type=str,
        default=None,
        help="Calibration file (default: cal_set.txt in data dir)",
    )
    parser.add_argument(
        "--cal_weight",
        type=float,
        default=0.9,
        help="Weight for misclassification in calibration loss (default 0.9)",
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

    # Class imbalance handling
    parser.add_argument(
        "--class_weight",
        type=float,
        default=None,
        help="Weight multiplier for minority class (success) loss. If None, auto-compute from data.",
    )
    parser.add_argument(
        "--oversample",
        action="store_true",
        help="Enable oversampling of minority class during training",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=0.0,
        help="Focal loss gamma (0 = disabled, 2.0 = typical). Focuses on hard examples.",
    )

    # Performance
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for faster training (PyTorch 2.0+)",
    )

    # Early stopping
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience (epochs without improvement). 0 = disabled. (default: system-specific)",
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=1e-6,
        help="Minimum improvement to reset patience counter",
    )

    return parser.parse_args()


def train_epoch(
    model: DualLyapunovNetwork,
    dataloader: DataLoader,
    loss_fn: LyapunovContrastiveLoss,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_clip: float = 0.0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_metrics = defaultdict(float)
    num_batches = 0

    for batch in dataloader:
        if len(batch) == 4:
            x_t_enc, x_tp1_enc, labels, is_last_pair = batch
            is_last_pair = is_last_pair.to(device, non_blocking=True)
        else:
            x_t_enc, x_tp1_enc, labels = batch
            is_last_pair = None
        # Use non_blocking for async GPU transfer (works with pin_memory)
        x_t_enc = x_t_enc.to(device, non_blocking=True)
        x_tp1_enc = x_tp1_enc.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass
        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
        losses = loss_fn(model, x_t_enc, x_tp1_enc, labels, is_last_pair=is_last_pair)

        # Backward pass
        losses["total_loss"].backward()

        # Gradient clipping
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Accumulate metrics
        for key, value in losses.items():
            total_metrics[key] += value.item()
        num_batches += 1

    # Average metrics
    return {k: v / num_batches for k, v in total_metrics.items()}


@torch.no_grad()
def validate(
    model: DualLyapunovNetwork,
    dataloader: DataLoader,
    loss_fn: LyapunovContrastiveLoss,
    device: str,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_metrics = defaultdict(float)
    num_batches = 0

    for batch in dataloader:
        if len(batch) == 4:
            x_t_enc, x_tp1_enc, labels, is_last_pair = batch
            is_last_pair = is_last_pair.to(device, non_blocking=True)
        else:
            x_t_enc, x_tp1_enc, labels = batch
            is_last_pair = None
        x_t_enc = x_t_enc.to(device, non_blocking=True)
        x_tp1_enc = x_tp1_enc.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        losses = loss_fn(model, x_t_enc, x_tp1_enc, labels, is_last_pair=is_last_pair)

        for key, value in losses.items():
            total_metrics[key] += value.item()
        num_batches += 1

    return {f"val_{k}": v / num_batches for k, v in total_metrics.items()}


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"Training Dual Lyapunov Network for {args.system}")
    print(f"{'='*60}\n")

    # Get system config
    config = get_system_config(args.system)
    print(f"System: {config.name}")
    print(f"  State dim: {config.state_dim}")
    print(f"  Input dim (encoded): {config.input_dim}")
    print(f"  Angle indices: {config.angle_indices}")

    # Apply system-specific defaults for None values
    if args.margin is None:
        args.margin = config.default_margin
    if args.lambda_contrastive is None:
        args.lambda_contrastive = config.default_lambda_contrastive
    if args.lr is None:
        args.lr = config.default_lr
    if args.weight_decay is None:
        args.weight_decay = config.default_weight_decay
    if args.patience is None:
        args.patience = config.default_patience
    if args.batch_size is None:
        args.batch_size = config.default_batch_size
    if args.epochs is None:
        args.epochs = config.default_epochs

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(args.output_dir) / args.system
    base_dir.mkdir(parents=True, exist_ok=True)
    output_dir = base_dir / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Create/update symlink to latest run
    latest_link = base_dir / "latest"
    if latest_link.is_symlink():
        latest_link.unlink()
    elif latest_link.exists():
        shutil.rmtree(latest_link)
    latest_link.symlink_to(output_dir.name, target_is_directory=True)
    print(f"Latest symlink: {latest_link} -> {output_dir.name}")

    # Create datasets
    print("\nLoading data...")
    num_train = args.num_train if args.num_train is not None else config.default_num_train
    train_dataset = LyapunovTrajectoryDataset(
        system_name=args.system,
        indices_file=args.indices_file,
        labels_file=args.labels_file,
        num_trajectories=num_train,
    )

    test_dataset = LyapunovTestDataset(system_name=args.system)

    # Extract attractor points (encoded terminal states per basin)
    attractor_points_0, attractor_points_1 = train_dataset.get_attractor_points()
    print(f"\nAttractor points: basin 0: {attractor_points_0.shape[0]}, basin 1: {attractor_points_1.shape[0]}")

    # Compute class distribution and weights
    labels_list = [pair[2] for pair in train_dataset.pairs]
    n_class_0 = sum(1 for l in labels_list if l == 0)  # failures
    n_class_1 = sum(1 for l in labels_list if l == 1)  # successes
    print(f"\nClass distribution:")
    print(f"  Class 0 (failure): {n_class_0} ({100*n_class_0/len(labels_list):.1f}%)")
    print(f"  Class 1 (success): {n_class_1} ({100*n_class_1/len(labels_list):.1f}%)")

    # Compute class weight for loss function
    if args.class_weight is not None:
        class_weight = args.class_weight
    else:
        # Auto-compute: inverse frequency ratio
        class_weight = n_class_0 / max(n_class_1, 1)
    print(f"  Class weight for minority: {class_weight:.2f}")

    # Create sampler for oversampling minority class
    sampler = None
    shuffle = True
    if args.oversample and n_class_1 > 0:
        # Assign sample weights inversely proportional to class frequency
        sample_weights = []
        weight_0 = 1.0
        weight_1 = n_class_0 / n_class_1  # upweight minority
        for label in labels_list:
            sample_weights.append(weight_1 if label == 1 else weight_0)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False  # Can't use shuffle with sampler
        print(f"  Oversampling enabled (minority weight: {weight_1:.2f}x)")

    # Create data loaders with optimized settings
    # Since data is pre-encoded, we can use more workers and persistent workers
    num_workers = 4 if args.device == "cuda" else 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(args.device == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,  # Larger batch for eval (no gradients)
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(args.device == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    # Create model
    hidden_sizes = args.hidden_sizes
    if hidden_sizes is None:
        hidden_sizes = DEFAULT_HIDDEN_SIZES.get(args.system, [64, 64, 64, 64])

    print(f"\nModel architecture:")
    print(f"  Hidden sizes: {hidden_sizes}")
    print(f"  Eps: {args.eps}")

    model = DualLyapunovNetwork(
        input_dim=config.input_dim,
        attractor_points_0=attractor_points_0,
        attractor_points_1=attractor_points_1,
        hidden_sizes=hidden_sizes,
        eps=args.eps,
    )
    model = model.to(args.device)

    # Optional: compile model for faster training (PyTorch 2.0+)
    # Note: neuromancer library has compatibility issues with torch.compile
    if args.compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        if hasattr(torch, "_dynamo"):
            torch._dynamo.config.capture_scalar_outputs = True
            torch._dynamo.config.suppress_errors = True  # Fall back to eager on errors
        model = torch.compile(model)
    elif args.compile:
        print("Warning: torch.compile not available (requires PyTorch 2.0+)")

    # Create loss function
    loss_fn = LyapunovContrastiveLoss(
        epsilon=args.epsilon,
        margin=args.margin,
        lambda_decrease=args.lambda_decrease,
        lambda_contrastive=args.lambda_contrastive,
        lambda_attractor=args.lambda_attractor,
        class_weight=class_weight,
        focal_gamma=args.focal_gamma,
    )

    print(f"\nLoss parameters:")
    print(f"  Epsilon (decrease): {args.epsilon}")
    print(f"  Margin (contrastive): {args.margin}")
    print(f"  Lambda decrease: {args.lambda_decrease}")
    print(f"  Lambda contrastive: {args.lambda_contrastive}")
    print(f"  Lambda attractor: {args.lambda_attractor}")
    print(f"  Class weight: {class_weight:.2f}")
    print(f"  Focal gamma: {args.focal_gamma}")

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    print(f"\nTraining parameters:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {args.device}")
    print(f"  Oversampling: {args.oversample}")
    print(f"  Early stopping: patience={args.patience}, min_delta={args.min_delta}")

    # Save full configuration
    config_dict = {
        "timestamp": timestamp,
        "system": args.system,
        "args": vars(args),
        "class_distribution": {
            "n_class_0": n_class_0,
            "n_class_1": n_class_1,
            "class_weight": class_weight,
        },
        "model": {
            "input_dim": config.input_dim,
            "hidden_sizes": hidden_sizes,
            "eps": args.eps,
            "n_attractors_0": attractor_points_0.shape[0],
            "n_attractors_1": attractor_points_1.shape[0],
        },
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"\nSaved config: {config_path}")

    # Training loop
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}")
    if args.lambda_attractor > 0:
        print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'V@Attr0':>10} {'V@Attr1':>10} {'Best':>6} {'LR':>10}")
        print("-" * 72)
    else:
        print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'Best':>6} {'LR':>10}")
        print("-" * 50)

    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in tqdm(range(args.epochs), desc="Training", unit="epoch"):
        train_metrics = train_epoch(
            model, train_loader, loss_fn, optimizer, args.device, args.grad_clip
        )
        val_metrics = validate(model, test_loader, loss_fn, args.device)

        scheduler.step()

        # Track best model with min_delta threshold
        current_loss = val_metrics["val_total_loss"]
        is_best = current_loss < (best_val_loss - args.min_delta)
        if is_best:
            best_val_loss = current_loss
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Log progress every epoch
        best_marker = "*" if is_best else ""
        if args.lambda_attractor > 0:
            # Show mean V at attractor proxies per basin (= loss_attractor_{0,1} before weighting)
            tqdm.write(
                f"{epoch+1:>6} {train_metrics['total_loss']:>12.6f} "
                f"{val_metrics['val_total_loss']:>12.6f} "
                f"{train_metrics['loss_attractor_0']:>10.6f} "
                f"{train_metrics['loss_attractor_1']:>10.6f} "
                f"{best_marker:>6} "
                f"{scheduler.get_last_lr()[0]:>10.6f}"
            )
        else:
            tqdm.write(
                f"{epoch+1:>6} {train_metrics['total_loss']:>12.6f} "
                f"{val_metrics['val_total_loss']:>12.6f} {best_marker:>6} "
                f"{scheduler.get_last_lr()[0]:>10.6f}"
            )

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                    "attractor_points_0": attractor_points_0,
                    "attractor_points_1": attractor_points_1,
                },
                checkpoint_path,
            )
            print(f"  Saved checkpoint: {checkpoint_path}")

        # Early stopping
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save final model
    final_model_path = output_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "config": {
                "system": args.system,
                "input_dim": config.input_dim,
                "hidden_sizes": hidden_sizes,
            },
            "attractor_points_0": attractor_points_0,
            "attractor_points_1": attractor_points_1,
        },
        final_model_path,
    )
    print(f"\nSaved best model: {final_model_path}")

    # Calibration (required for evaluation)
    print(f"\n{'='*60}")
    print("Running Calibration")
    print(f"{'='*60}")

    # Determine calibration file
    if args.cal_file:
        cal_file = Path(args.cal_file)
    else:
        data_dir = get_data_dir(args.system)
        cal_file = data_dir / "cal_set.txt"

    cal_result = None
    cal_output_dir = None
    if cal_file.exists():
        cal_result = calibrate(
            model=model,
            cal_file=cal_file,
            system_name=args.system,
            n_quantiles=args.n_quantiles,
            n_delta=args.n_delta,
            w=args.cal_weight,
            device=args.device,
            verbose=True,
        )

        # Create parameter-named subdirectory for calibration results
        cal_dir_name = get_cal_dir_name(args.cal_weight, args.n_quantiles, args.n_delta)
        cal_output_dir = output_dir / cal_dir_name
        cal_output_dir.mkdir(parents=True, exist_ok=True)

        # Save calibration results
        cal_path = cal_output_dir / "calibration.json"
        cal_result.save(cal_path)
        print(f"Saved calibration: {cal_path}")
    else:
        print(f"Calibration file not found: {cal_file}")
        print("Skipping calibration and evaluation.")

    # Final evaluation (requires calibration)
    if cal_result is not None and cal_output_dir is not None:
        print(f"\n{'='*60}")
        print("Final Evaluation on Test Set")
        print(f"{'='*60}")

        evaluator = LyapunovEvaluator(
            model,
            c1=cal_result.c1,
            c2=cal_result.c2,
            delta=cal_result.delta,
            device=args.device,
        )
        test_metrics = evaluator.evaluate_all(test_loader)
        print_evaluation_results(test_metrics)

        # Save metrics to calibration subdirectory
        metrics_path = cal_output_dir / "test_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
