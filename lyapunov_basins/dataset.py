"""Dataset classes for loading trajectory data."""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .systems import SystemConfig, StateEncoder, get_system_config, get_data_dir


class LyapunovTrajectoryDataset(Dataset):
    """
    Dataset for training Lyapunov functions from trajectory data.

    Loads trajectories and creates consecutive state pairs (x_t, x_{t+1})
    with basin labels for the Lyapunov decrease condition.

    All data is pre-encoded during initialization for maximum training speed.
    """

    def __init__(
        self,
        system_name: str,
        indices_file: str = "shuffled_indices_0.txt",
        labels_file: str = "shuffled_labels_0.txt",
        num_trajectories: Optional[int] = None,
        data_dir: Optional[Path] = None,
    ):
        """
        Initialize dataset.

        Args:
            system_name: Name of the system ('pendulum', 'cartpole', etc.)
            indices_file: Filename for trajectory indices in train_test_splits/
            labels_file: Filename for trajectory labels in train_test_splits/
            num_trajectories: Number of trajectories to use (None = all)
            data_dir: Override default data directory
        """
        self.system_name = system_name
        self.config = get_system_config(system_name)
        self.encoder = StateEncoder(self.config)

        if data_dir is None:
            data_dir = get_data_dir(system_name)
        self.data_dir = Path(data_dir)

        # Load trajectory indices and labels
        splits_dir = self.data_dir / "train_test_splits"
        indices_path = splits_dir / indices_file
        labels_path = splits_dir / labels_file

        with open(indices_path, "r") as f:
            trajectory_files = [line.strip() for line in f.readlines()]

        with open(labels_path, "r") as f:
            labels = [int(line.strip()) for line in f.readlines()]

        # Limit number of trajectories if specified
        if num_trajectories is not None:
            trajectory_files = trajectory_files[:num_trajectories]
            labels = labels[:num_trajectories]

        # Load all trajectories and create consecutive pairs
        x_t_list: List[Tensor] = []
        x_tp1_list: List[Tensor] = []
        labels_list: List[int] = []
        is_last_pair_list: List[bool] = []
        trajectories_dir = self.data_dir / "trajectories"

        for traj_file, label in zip(trajectory_files, labels):
            traj_path = trajectories_dir / traj_file
            traj = self._load_trajectory(traj_path)

            if len(traj) < 2:
                # Skip trajectories with less than 2 states
                continue

            # Create consecutive pairs
            num_pairs = len(traj) - 1
            for t in range(num_pairs):
                x_t_list.append(traj[t])
                x_tp1_list.append(traj[t + 1])
                labels_list.append(label)
                is_last_pair_list.append(t == num_pairs - 1)

        # Stack into tensors and pre-encode (vectorized)
        print(f"Pre-encoding {len(x_t_list)} state pairs...")
        x_t_raw = torch.stack(x_t_list)  # (N, state_dim)
        x_tp1_raw = torch.stack(x_tp1_list)  # (N, state_dim)

        # Vectorized encoding - much faster than per-sample
        self.x_t_enc = self.encoder.encode(x_t_raw)  # (N, input_dim)
        self.x_tp1_enc = self.encoder.encode(x_tp1_raw)  # (N, input_dim)
        self.labels = torch.tensor(labels_list, dtype=torch.long)  # (N,)
        # Marks the last pair per trajectory: x_tp1 of this pair is the terminal
        # state, used as an attractor proxy by the attractor loss.
        self.is_last_pair = torch.tensor(is_last_pair_list, dtype=torch.bool)  # (N,)

        # Store raw pairs for computing class weights (used by training script)
        self.pairs = [(None, None, label) for label in labels_list]

        n_attractor = int(self.is_last_pair.sum().item())
        n_att_0 = int((self.is_last_pair & (self.labels == 0)).sum().item())
        n_att_1 = int((self.is_last_pair & (self.labels == 1)).sum().item())
        print(f"Loaded {len(self.labels)} state pairs from {len(trajectory_files)} trajectories")
        print(f"  Attractor proxies (terminal states): {n_attractor} (basin 0: {n_att_0}, basin 1: {n_att_1})")

    def get_attractor_points(self) -> Tuple[Tensor, Tensor]:
        """
        Extract encoded terminal states (attractor proxies) per basin.

        Returns:
            attractor_points_0: (K0, input_dim) terminal states for basin 0
            attractor_points_1: (K1, input_dim) terminal states for basin 1
        """
        mask_0 = self.is_last_pair & (self.labels == 0)
        mask_1 = self.is_last_pair & (self.labels == 1)
        return self.x_tp1_enc[mask_0], self.x_tp1_enc[mask_1]

    def _load_trajectory(self, path: Path) -> Tensor:
        """Load a trajectory file as a tensor."""
        data = np.loadtxt(path, delimiter=",")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return torch.from_numpy(data).float()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Get a state pair.

        Returns:
            x_t_enc: Encoded current state (input_dim,)
            x_tp1_enc: Encoded next state (input_dim,)
            label: Basin label (scalar tensor)
            is_last_pair: Whether this is the last pair in a trajectory (scalar bool).
                         When True, x_tp1_enc is the trajectory's terminal state
                         (attractor proxy).
        """
        return self.x_t_enc[idx], self.x_tp1_enc[idx], self.labels[idx], self.is_last_pair[idx]


class LyapunovTestDataset(Dataset):
    """
    Dataset for evaluating Lyapunov functions on the test set.

    Loads test_set.txt which contains initial_state + final_state + label per line.
    All data is pre-encoded during initialization for maximum evaluation speed.
    """

    def __init__(
        self,
        system_name: str,
        data_dir: Optional[Path] = None,
    ):
        """
        Initialize test dataset.

        Args:
            system_name: Name of the system
            data_dir: Override default data directory
        """
        self.system_name = system_name
        self.config = get_system_config(system_name)
        self.encoder = StateEncoder(self.config)

        if data_dir is None:
            data_dir = get_data_dir(system_name)
        self.data_dir = Path(data_dir)

        # Load test set
        test_path = self.data_dir / "test_set.txt"
        data = np.loadtxt(test_path, delimiter=",")

        state_dim = self.config.state_dim

        # Parse columns: init_state (state_dim) + final_state (state_dim) + label (1)
        x_init_raw = torch.from_numpy(data[:, :state_dim]).float()
        x_final_raw = torch.from_numpy(data[:, state_dim:2*state_dim]).float()
        self.labels = torch.from_numpy(data[:, -1]).long()

        # Pre-encode all data (vectorized)
        print(f"Pre-encoding {len(self.labels)} test samples...")
        self.x_init_enc = self.encoder.encode(x_init_raw)  # (N, input_dim)
        self.x_final_enc = self.encoder.encode(x_final_raw)  # (N, input_dim)

        print(f"Loaded {len(self.labels)} test samples")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Get a test sample.

        Returns:
            x_init_enc: Encoded initial state (input_dim,)
            x_final_enc: Encoded final state (input_dim,)
            label: Ground truth basin label (scalar tensor)
        """
        return self.x_init_enc[idx], self.x_final_enc[idx], self.labels[idx]


class LyapunovStateDataset(Dataset):
    """
    Simple dataset of individual states with labels.

    Useful for classification evaluation where we only need
    to compare V_0(x) vs V_1(x) on individual states.
    All data is pre-encoded during initialization for maximum speed.
    """

    def __init__(
        self,
        system_name: str,
        indices_file: str = "shuffled_indices_0.txt",
        labels_file: str = "shuffled_labels_0.txt",
        num_trajectories: Optional[int] = None,
        data_dir: Optional[Path] = None,
    ):
        """
        Initialize dataset with individual states (not pairs).
        """
        self.system_name = system_name
        self.config = get_system_config(system_name)
        self.encoder = StateEncoder(self.config)

        if data_dir is None:
            data_dir = get_data_dir(system_name)
        self.data_dir = Path(data_dir)

        # Load trajectory indices and labels
        splits_dir = self.data_dir / "train_test_splits"
        indices_path = splits_dir / indices_file
        labels_path = splits_dir / labels_file

        with open(indices_path, "r") as f:
            trajectory_files = [line.strip() for line in f.readlines()]

        with open(labels_path, "r") as f:
            labels = [int(line.strip()) for line in f.readlines()]

        # Limit number of trajectories if specified
        if num_trajectories is not None:
            trajectory_files = trajectory_files[:num_trajectories]
            labels = labels[:num_trajectories]

        # Load all states
        states_list: List[Tensor] = []
        labels_list: List[int] = []
        trajectories_dir = self.data_dir / "trajectories"

        for traj_file, label in zip(trajectory_files, labels):
            traj_path = trajectories_dir / traj_file
            traj = self._load_trajectory(traj_path)

            for t in range(len(traj)):
                states_list.append(traj[t])
                labels_list.append(label)

        # Stack and pre-encode (vectorized)
        print(f"Pre-encoding {len(states_list)} states...")
        x_raw = torch.stack(states_list)  # (N, state_dim)
        self.x_enc = self.encoder.encode(x_raw)  # (N, input_dim)
        self.labels = torch.tensor(labels_list, dtype=torch.long)  # (N,)

        print(f"Loaded {len(self.labels)} states from {len(trajectory_files)} trajectories")

    def _load_trajectory(self, path: Path) -> Tensor:
        """Load a trajectory file as a tensor."""
        data = np.loadtxt(path, delimiter=",")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return torch.from_numpy(data).float()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Get a state.

        Returns:
            x_enc: Encoded state (input_dim,)
            label: Basin label (scalar tensor)
        """
        return self.x_enc[idx], self.labels[idx]
