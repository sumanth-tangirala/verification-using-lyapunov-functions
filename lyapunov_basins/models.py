"""Neural Lyapunov network models using NeuroMANCER."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from neuromancer.modules import blocks


# Default architectures scaled to input dimension
DEFAULT_HIDDEN_SIZES: Dict[str, List[int]] = {
    "pendulum": [32, 32, 32],                # input_dim=3
    "cartpole": [128, 128, 128, 128],        # input_dim=5
    "quadrotor2d": [192, 192, 192, 192, 192],  # input_dim=7
    "quadrotor3d": [256, 256, 256, 256, 256],  # input_dim=13
}


class SmoothedReLU(nn.Module):
    """
    ReLU with a quadratic region in [0,d] (Rectified Huber Unit).

    Makes the Lyapunov function continuously differentiable.
    Reimplemented from neuromancer.modules.activations.SmoothedReLU.
    See https://arxiv.org/pdf/2001.06116.pdf
    """

    def __init__(self, d: float = 1.0, tune_d: bool = True):
        super().__init__()
        self.d = nn.Parameter(torch.tensor(d), requires_grad=tune_d)

    def forward(self, x: Tensor) -> Tensor:
        alpha = 1.0 / F.softplus(self.d)
        beta = -F.softplus(self.d) / 2
        return torch.max(
            torch.clamp(
                torch.sign(x) * torch.div(alpha, 2.0) * x ** 2,
                min=0, max=-beta.item(),
            ),
            x + beta,
        )


class AttractorPosDef(nn.Module):
    """
    Positive-definite Lyapunov wrapper with attractor-set zero-point.

    Instead of shifting g(x) by g(0) (as in neuromancer's PosDef),
    this shifts by max_{a in A} g(a), where A is the set of encoded
    attractor points (trajectory terminal states) for this basin.

        V(x) = smReLU(g(x) - max_{a in A} g(a))

    Properties:
        - V(a) = 0 for all a in A (by construction)
        - V(x) >= 0 everywhere (smReLU is non-negative)
    """

    def __init__(
        self,
        g: nn.Module,
        attractor_points: Tensor,
        d: float = 1.0,
    ):
        """
        Args:
            g: ICNN network (e.g., blocks.InputConvexNN)
            attractor_points: Encoded attractor states (K, input_dim)
            d: Initial d parameter for SmoothedReLU
        """
        super().__init__()
        self.g = g
        self.in_features = g.in_features
        self.out_features = g.out_features
        self.smReLU = SmoothedReLU(d)
        self.register_buffer("attractor_points", attractor_points.clone())

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute V(x) = smReLU(g(x) - max_{a in A} g(a)).

        Args:
            x: Input states (N, input_dim)

        Returns:
            Lyapunov values (N, 1), non-negative, zero at attractor points
        """
        g_x = self.g(x)                              # (N, 1)
        g_attractors = self.g(self.attractor_points)  # (K, 1)
        max_g_a = g_attractors.max()                  # scalar
        return self.smReLU(g_x - max_g_a)


class DualLyapunovNetwork(nn.Module):
    """
    Dual Lyapunov network for two-basin classification.

    Contains two separate Lyapunov function candidates (V_0, V_1),
    each implemented as ICNN + AttractorPosDef to ensure:
    - V_i(a) = 0 for all attractor points a in basin i
    - V_i(x) >= 0 everywhere

    Basin classification: predict basin 0 if V_0(x) < V_1(x), else basin 1.
    """

    def __init__(
        self,
        input_dim: int,
        attractor_points_0: Tensor,
        attractor_points_1: Tensor,
        hidden_sizes: Optional[List[int]] = None,
        eps: float = 0.01,
    ):
        """
        Initialize dual Lyapunov network.

        Args:
            input_dim: Dimension of encoded input state
            attractor_points_0: Encoded attractor states for basin 0 (K0, input_dim)
            attractor_points_1: Encoded attractor states for basin 1 (K1, input_dim)
            hidden_sizes: List of hidden layer sizes for ICNN.
                         If None, uses default based on input_dim.
            eps: Unused, kept for backward compatibility.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes if hidden_sizes is not None else [64, 64, 64, 64]
        self.eps = eps

        # V_0: ICNN + AttractorPosDef for basin 0
        icnn_0 = blocks.InputConvexNN(input_dim, 1, hsizes=self.hidden_sizes)
        self.V_0 = AttractorPosDef(icnn_0, attractor_points_0)

        # V_1: ICNN + AttractorPosDef for basin 1
        icnn_1 = blocks.InputConvexNN(input_dim, 1, hsizes=self.hidden_sizes)
        self.V_1 = AttractorPosDef(icnn_1, attractor_points_1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute Lyapunov values for both basins.

        Args:
            x: Encoded states (..., input_dim)

        Returns:
            V_0: Lyapunov values for basin 0 (..., 1)
            V_1: Lyapunov values for basin 1 (..., 1)
        """
        V_0 = self.V_0(x)
        V_1 = self.V_1(x)
        return V_0, V_1

    def predict_basin(self, x: Tensor) -> Tensor:
        """
        Predict basin membership by comparing V_0(x) vs V_1(x).

        Lower Lyapunov value indicates the state belongs to that basin.

        Args:
            x: Encoded states (..., input_dim)

        Returns:
            predictions: Basin labels (...,) with values 0 or 1
        """
        V_0, V_1 = self.forward(x)
        # Return 0 if V_0 < V_1 (state in basin 0), else 1
        return (V_1 < V_0).long().squeeze(-1)

    def get_lyapunov_values(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Get Lyapunov values as a dictionary (useful for logging).

        Args:
            x: Encoded states (..., input_dim)

        Returns:
            Dictionary with 'V_0' and 'V_1' tensors
        """
        V_0, V_1 = self.forward(x)
        return {"V_0": V_0, "V_1": V_1}

    def extra_repr(self) -> str:
        n_att_0 = self.V_0.attractor_points.shape[0]
        n_att_1 = self.V_1.attractor_points.shape[0]
        return (
            f"input_dim={self.input_dim}, hidden_sizes={self.hidden_sizes}, "
            f"attractors_0={n_att_0}, attractors_1={n_att_1}"
        )


def create_model(
    system_name: str,
    input_dim: int,
    attractor_points_0: Tensor,
    attractor_points_1: Tensor,
    hidden_sizes: Optional[List[int]] = None,
    eps: float = 0.01,
) -> DualLyapunovNetwork:
    """
    Create a DualLyapunovNetwork with system-appropriate defaults.

    Args:
        system_name: Name of the system ('pendulum', 'cartpole', etc.)
        input_dim: Dimension of encoded input state
        attractor_points_0: Encoded attractor states for basin 0
        attractor_points_1: Encoded attractor states for basin 1
        hidden_sizes: Override default hidden sizes
        eps: Unused, kept for backward compatibility

    Returns:
        Initialized DualLyapunovNetwork
    """
    if hidden_sizes is None:
        hidden_sizes = DEFAULT_HIDDEN_SIZES.get(system_name, [64, 64, 64, 64])

    return DualLyapunovNetwork(
        input_dim=input_dim,
        attractor_points_0=attractor_points_0,
        attractor_points_1=attractor_points_1,
        hidden_sizes=hidden_sizes,
        eps=eps,
    )
