"""System configurations and state encoding for Lyapunov training."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv
import torch
from torch import Tensor


# Load environment variables from .env file
load_dotenv()

DATA_BASE_DIR = Path(os.environ.get("DATA_DIR", "/common/users/shared/pracsys/genMoPlan/data_trajectories"))


@dataclass
class SystemConfig:
    """Configuration for a dynamical system."""

    name: str
    dataset_name: str
    state_dim: int
    input_dim: int  # After encoding (sin/cos expansion)
    angle_indices: List[int]  # Indices of angle states to encode as sin/cos
    state_names: List[str]
    achieved_bounds: List[Tuple[float, float]]  # (min, max) for each state
    default_hidden_sizes: List[int]
    default_num_train: Optional[int] = None  # Default number of training samples

    # Training hyperparameters (system-specific defaults)
    default_margin: float = 1.0
    default_lambda_contrastive: float = 1.0
    default_lr: float = 0.0005
    default_weight_decay: float = 0.0001
    default_patience: int = 20
    default_batch_size: int = 256
    default_epochs: int = 200

    @property
    def euclidean_indices(self) -> List[int]:
        """Indices of non-angle (Euclidean) states that need normalization."""
        return [i for i in range(self.state_dim) if i not in self.angle_indices]


# System configurations (bounds will be loaded from dataset_description.json)
SYSTEM_CONFIGS = {
    "pendulum": SystemConfig(
        name="pendulum",
        dataset_name="pendulum_lqr_50k",
        state_dim=2,
        input_dim=3,  # sin(theta), cos(theta), theta_dot_norm
        angle_indices=[0],  # theta
        state_names=["theta", "theta_dot"],
        achieved_bounds=[],  # Will be loaded
        default_hidden_sizes=[32, 32, 32],
        default_num_train=1000,
        # Training defaults
        default_margin=1.0,
        default_lambda_contrastive=1.0,
        default_lr=0.0005,
        default_weight_decay=0.001,
        default_patience=10,
        default_batch_size=256,
        default_epochs=200,
    ),
    "cartpole": SystemConfig(
        name="cartpole",
        dataset_name="cartpole_pybullet",
        state_dim=4,
        input_dim=5,  # x_norm, sin(theta), cos(theta), x_dot_norm, theta_dot_norm
        angle_indices=[1],  # theta
        state_names=["x", "theta", "x_dot", "theta_dot"],
        achieved_bounds=[],  # Will be loaded
        default_hidden_sizes=[128, 128, 128, 128],
        default_num_train=2000,
        # Training defaults
        default_margin=5.0,
        default_lambda_contrastive=5.0,
        default_lr=0.0005,
        default_weight_decay=0.0001,
        default_patience=10,
        default_batch_size=128,
        default_epochs=300,
    ),
    "quadrotor2d": SystemConfig(
        name="quadrotor2d",
        dataset_name="quadrotor2D_rl",
        state_dim=6,
        input_dim=7,  # x, z, sin(theta), cos(theta), x_dot, z_dot, theta_dot (all normalized except sin/cos)
        angle_indices=[2],  # theta
        state_names=["x", "z", "theta", "x_dot", "z_dot", "theta_dot"],
        achieved_bounds=[],  # Will be loaded
        default_hidden_sizes=[192, 192, 192, 192, 192],
        default_num_train=4000,
        # Training defaults
        default_margin=12.0,
        default_lambda_contrastive=16.0,
        default_lr=0.0005,
        default_weight_decay=0.00005,
        default_patience=25,
        default_batch_size=256,
        default_epochs=500,
    ),
    "quadrotor3d": SystemConfig(
        name="quadrotor3d",
        dataset_name="quadrotor3D_lqr",
        state_dim=13,
        input_dim=13,  # No angle expansion (quaternions already present)
        angle_indices=[],  # qw, qx, qy, qz are already in proper form
        state_names=["x", "y", "z", "qw", "qx", "qy", "qz", "x_dot", "y_dot", "z_dot", "p", "q", "r"],
        achieved_bounds=[],  # Will be loaded
        default_hidden_sizes=[256, 256, 256, 256, 256],
        default_num_train=6000,
        # Training defaults
        default_margin=12.0,
        default_lambda_contrastive=12.0,
        default_lr=0.001,
        default_weight_decay=0.00002,
        default_patience=10,
        default_batch_size=256,
        default_epochs=500,
    ),
}

# Indices that should NOT be normalized (quaternion components for quadrotor3d)
QUATERNION_INDICES = {
    "quadrotor3d": [3, 4, 5, 6],  # qw, qx, qy, qz
}


def load_achieved_bounds(system_name: str) -> List[Tuple[float, float]]:
    """Load achieved_bounds from dataset_description.json."""
    config = SYSTEM_CONFIGS[system_name]
    desc_path = DATA_BASE_DIR / config.dataset_name / "dataset_description.json"

    with open(desc_path, "r") as f:
        desc = json.load(f)

    achieved_bounds = desc["achieved_bounds"]
    bounds = []
    for state_name in config.state_names:
        bound = achieved_bounds[state_name]
        bounds.append((bound["min"], bound["max"]))

    return bounds


def get_system_config(system_name: str) -> SystemConfig:
    """Get system configuration with loaded bounds."""
    if system_name not in SYSTEM_CONFIGS:
        raise ValueError(f"Unknown system: {system_name}. Available: {list(SYSTEM_CONFIGS.keys())}")

    config = SYSTEM_CONFIGS[system_name]

    # Load bounds if not already loaded
    if not config.achieved_bounds:
        config.achieved_bounds = load_achieved_bounds(system_name)

    return config


class StateEncoder:
    """Encodes raw states for neural network input.

    Pipeline:
    1. Normalize Euclidean dimensions to [-1, 1] using achieved_bounds
    2. Convert angle dimensions to (sin, cos) pairs
    3. Quaternions (for quadrotor3d) are passed through unchanged
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.system_name = config.name

        # Precompute normalization parameters for Euclidean dimensions
        euclidean_indices = config.euclidean_indices

        # For quadrotor3d, exclude quaternion indices from normalization
        if self.system_name == "quadrotor3d":
            quat_indices = QUATERNION_INDICES["quadrotor3d"]
            euclidean_indices = [i for i in euclidean_indices if i not in quat_indices]

        self.euclidean_indices = euclidean_indices
        self.angle_indices = config.angle_indices

        if euclidean_indices:
            self._mins = torch.tensor(
                [config.achieved_bounds[i][0] for i in euclidean_indices],
                dtype=torch.float32
            )
            self._maxs = torch.tensor(
                [config.achieved_bounds[i][1] for i in euclidean_indices],
                dtype=torch.float32
            )
            self._ranges = self._maxs - self._mins
        else:
            self._mins = None
            self._maxs = None
            self._ranges = None

    def _ensure_device(self, x: Tensor) -> None:
        """Move precomputed tensors to same device as input."""
        if self._mins is not None and self._mins.device != x.device:
            self._mins = self._mins.to(x.device, dtype=x.dtype)
            self._maxs = self._maxs.to(x.device, dtype=x.dtype)
            self._ranges = self._ranges.to(x.device, dtype=x.dtype)

    def encode(self, x: Tensor) -> Tensor:
        """
        Encode raw state for network input.

        Args:
            x: Raw states (..., state_dim)

        Returns:
            x_enc: Encoded states (..., input_dim)
        """
        self._ensure_device(x)

        outputs = []

        for i in range(self.config.state_dim):
            component = x[..., i:i+1]

            if i in self.angle_indices:
                # Angle: convert to sin/cos
                outputs.append(torch.sin(component))
                outputs.append(torch.cos(component))
            elif i in self.euclidean_indices:
                # Euclidean: normalize to [-1, 1]
                idx_in_euclidean = self.euclidean_indices.index(i)
                min_val = self._mins[idx_in_euclidean]
                range_val = self._ranges[idx_in_euclidean]
                normalized = 2 * (component - min_val) / range_val - 1
                outputs.append(normalized)
            else:
                # Quaternion or other: pass through unchanged
                outputs.append(component)

        return torch.cat(outputs, dim=-1)


def encode_state(x: Tensor, config: SystemConfig) -> Tensor:
    """
    Convenience function to encode states.

    Args:
        x: Raw states (..., state_dim)
        config: System configuration

    Returns:
        x_enc: Encoded states (..., input_dim)
    """
    encoder = StateEncoder(config)
    return encoder.encode(x)


def get_data_dir(system_name: str) -> Path:
    """Get the data directory for a system."""
    config = SYSTEM_CONFIGS[system_name]
    return DATA_BASE_DIR / config.dataset_name
