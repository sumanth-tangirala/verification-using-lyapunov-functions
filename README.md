# Neural Lyapunov Basins

Learning dual Lyapunov functions for classifying dynamical system trajectories into stability basins.

## Overview

Given a dynamical system where trajectories can converge to one of two outcomes (e.g., success or failure), this project trains neural networks that predict which basin of attraction a given state belongs to. The approach uses:

- **Dual Lyapunov Functions**: Two neural networks V1(x) and V2(x) that decrease along trajectories heading toward their respective basins
- **Input Convex Neural Networks (ICNN)**: Ensures the learned functions have desirable convexity properties
- **Attractor-Anchored Positive Definite Layers**: Guarantees V(attractor) = 0 and V(x) > 0 elsewhere
- **Contrastive Learning**: Separates the two basins in the learned Lyapunov space
- **Calibrated Classification**: Grid search for optimal decision thresholds with uncertainty quantification

## Installation

**Using conda (recommended):**
```bash
conda env create -f environment.yml
conda activate lyapunov_nn
```

**Using pip:**
```bash
pip install torch neuromancer python-dotenv cvxpy wandb pytorch-lightning
```

## Configuration

Copy the example environment file and update paths:
```bash
cp .env.example .env
```

Edit `.env` to set:
- `DATA_DIR`: Path to trajectory data directory
- `OUTPUT_DIR`: Path for saving model outputs

## Usage

### Training

```bash
# Train on pendulum system
python train_lyapunov_basins.py --system pendulum --num_train 1000 --epochs 200

# Train on cartpole with custom settings
python train_lyapunov_basins.py --system cartpole \
    --num_train 2000 \
    --epochs 300 \
    --lr 0.0005 \
    --margin 5.0 \
    --lambda_contrastive 5.0

# Train on quadrotor2d
python train_lyapunov_basins.py --system quadrotor2d --num_train 4000 --epochs 500
```

### Re-calibration

After training, you can re-calibrate thresholds without retraining:
```bash
python recalibrate.py --run_dir outputs/pendulum/latest --cal_weight 0.5
python recalibrate.py --run_dir outputs/cartpole/latest --cal_weight 0.3 --n_quantiles 30
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--system` | System name (pendulum, cartpole, quadrotor2d, quadrotor3d) | required |
| `--num_train` | Number of training trajectories | system-specific |
| `--epochs` | Training epochs | system-specific |
| `--lr` | Learning rate | system-specific |
| `--margin` | Contrastive loss margin | system-specific |
| `--lambda_contrastive` | Weight for contrastive loss | system-specific |
| `--oversample` | Oversample minority class | False |
| `--patience` | Early stopping patience | system-specific |

## Supported Systems

| System | State Variables | Description |
|--------|----------------|-------------|
| `pendulum` | θ, θ̇ | Simple pendulum with LQR controller |
| `cartpole` | x, θ, ẋ, θ̇ | Cart-pole balancing (PyBullet) |
| `quadrotor2d` | x, z, θ, ẋ, ż, θ̇ | Planar quadrotor with RL controller |
| `quadrotor3d` | x, y, z, qw, qx, qy, qz, ẋ, ẏ, ż, p, q, r | Full 3D quadrotor with LQR |

## Output Structure

Training creates timestamped run directories:
```
outputs/
└── pendulum/
    ├── run_20260209_143052/
    │   ├── config.json          # Training configuration
    │   ├── best_model.pt        # Best model checkpoint
    │   ├── checkpoint_epoch_*.pt # Periodic checkpoints
    │   └── cal_w0.9_nq20_nd10/  # Calibration results
    │       ├── calibration.json
    │       └── test_metrics.json
    └── latest -> run_20260209_143052  # Symlink to latest run
```

## Method

1. **State Encoding**: Raw states are encoded for neural network input:
   - Angles → (sin, cos) pairs
   - Other states → normalized to [-1, 1]
   - Quaternions → passed through unchanged

2. **Dual Network**: Two parallel networks each produce a Lyapunov-like function:
   - ICNN layers ensure input convexity
   - AttractorPosDef layer anchors V=0 at learned attractor points

3. **Loss Function**:
   - **Decrease loss**: V(x_{t+1}) < V(x_t) - ε along trajectories
   - **Contrastive loss**: Separates V1 vs V2 values between basins

4. **Calibration**: Find thresholds (c1, c2, δ) for 3-way classification:
   - Basin 0: V1 < c1 and V1 < V2 - δ
   - Basin 1: V2 < c2 and V2 < V1 - δ
   - Uncertain: otherwise

## References

- Manek & Kolter, "Learning Stable Deep Dynamics Models" (NeurIPS 2019)
- Amos et al., "Input Convex Neural Networks" (ICML 2017)
- Richards et al., "The Lyapunov Neural Network" (IEEE L-CSS 2018)
