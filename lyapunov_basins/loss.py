"""Loss functions for dual Lyapunov network training."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .models import DualLyapunovNetwork


class LyapunovContrastiveLoss(nn.Module):
    """
    Combined loss for training dual Lyapunov functions.

    Loss components:
    1. Lyapunov Decrease Loss: Ensures V decreases along trajectories within each basin
       L_decrease = E[ReLU(V(x_{t+1}) - V(x_t) + epsilon)]

    2. Contrastive Loss: Ensures V_i is high for states from the other basin
       L_contrastive_0 = E_{x ~ basin_1}[ReLU(margin - V_0(x))]
       L_contrastive_1 = E_{x ~ basin_0}[ReLU(margin - V_1(x))]

    3. Attractor Loss: Penalizes V_i(x_final) for trajectories in basin i,
       driving the Lyapunov function toward zero at the observed attractor.
       L_attractor = mean(V_0(x_final) | basin 0) + class_weight * mean(V_1(x_final) | basin 1)

    Total loss = lambda_decrease * L_decrease + lambda_contrastive * L_contrastive
                 + lambda_attractor * L_attractor

    Supports class weighting to handle imbalanced datasets and focal loss
    to focus on hard examples.
    """

    def __init__(
        self,
        epsilon: float = 0.01,
        margin: float = 1.0,
        lambda_decrease: float = 1.0,
        lambda_contrastive: float = 1.0,
        lambda_attractor: float = 0.0,
        class_weight: float = 1.0,
        focal_gamma: float = 0.0,
    ):
        """
        Initialize loss function.

        Args:
            epsilon: Margin for Lyapunov decrease condition.
                    V(x_{t+1}) - V(x_t) < -epsilon
            margin: Margin for contrastive loss.
                    States from other basin should have V > margin
            lambda_decrease: Weight for decrease loss
            lambda_contrastive: Weight for contrastive loss
            lambda_attractor: Weight for attractor loss (0 = disabled)
            class_weight: Weight multiplier for minority class (class 1) losses.
                         Use > 1.0 to upweight minority class.
            focal_gamma: Focal loss gamma for hard example mining.
                        0 = disabled, 2.0 = typical value.
        """
        super().__init__()
        self.epsilon = epsilon
        self.margin = margin
        self.lambda_decrease = lambda_decrease
        self.lambda_contrastive = lambda_contrastive
        self.lambda_attractor = lambda_attractor
        self.class_weight = class_weight
        self.focal_gamma = focal_gamma

    def forward(
        self,
        model: DualLyapunovNetwork,
        x_t: Tensor,
        x_tp1: Tensor,
        labels: Tensor,
        is_last_pair: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute combined loss.

        Args:
            model: DualLyapunovNetwork
            x_t: Current encoded states (N, input_dim)
            x_tp1: Next encoded states (N, input_dim)
            labels: Basin labels (N,) with values 0 or 1
            is_last_pair: Boolean mask indicating last pair per trajectory (N,).
                         When True, x_tp1 is that trajectory's terminal state
                         (attractor proxy). None disables attractor loss.

        Returns:
            Dictionary with total loss and component losses
        """
        # Compute Lyapunov values
        V_0_t, V_1_t = model(x_t)      # (N, 1), (N, 1)
        V_0_tp1, V_1_tp1 = model(x_tp1)

        # Create masks for each basin
        mask_0 = (labels == 0).float().unsqueeze(-1)  # (N, 1)
        mask_1 = (labels == 1).float().unsqueeze(-1)  # (N, 1)

        n_basin_0 = mask_0.sum() + 1e-8
        n_basin_1 = mask_1.sum() + 1e-8

        # 1. Lyapunov Decrease Loss
        # For basin 0: V_0(x_{t+1}) - V_0(x_t) < -epsilon
        # For basin 1: V_1(x_{t+1}) - V_1(x_t) < -epsilon
        decrease_0 = F.relu((V_0_tp1 - V_0_t) + self.epsilon)  # (N, 1)
        decrease_1 = F.relu((V_1_tp1 - V_1_t) + self.epsilon)  # (N, 1)

        # Apply focal loss weighting if enabled
        if self.focal_gamma > 0:
            # Normalize violations to [0, 1] range for focal weighting
            # Higher violation = easier example (model is confident but wrong)
            # We want to focus on smaller violations (harder examples)
            focal_weight_0 = (1 - torch.exp(-decrease_0)) ** self.focal_gamma
            focal_weight_1 = (1 - torch.exp(-decrease_1)) ** self.focal_gamma
            decrease_0 = decrease_0 * focal_weight_0
            decrease_1 = decrease_1 * focal_weight_1

        # Apply to respective basins and compute mean
        loss_decrease_0 = (decrease_0 * mask_0).sum() / n_basin_0
        loss_decrease_1 = (decrease_1 * mask_1).sum() / n_basin_1
        # Apply class weight to minority class (basin 1 = success)
        loss_decrease = loss_decrease_0 + self.class_weight * loss_decrease_1

        # 2. Contrastive Loss
        # For states from basin 1: V_0 should be high (at least margin)
        # For states from basin 0: V_1 should be high (at least margin)
        contrastive_0 = F.relu(self.margin - V_0_t)  # Penalize low V_0
        contrastive_1 = F.relu(self.margin - V_1_t)  # Penalize low V_1

        # Apply focal loss weighting if enabled
        if self.focal_gamma > 0:
            # Normalize to margin range
            focal_weight_c0 = (contrastive_0 / (self.margin + 1e-8)) ** self.focal_gamma
            focal_weight_c1 = (contrastive_1 / (self.margin + 1e-8)) ** self.focal_gamma
            contrastive_0 = contrastive_0 * focal_weight_c0
            contrastive_1 = contrastive_1 * focal_weight_c1

        loss_contrastive_0 = (contrastive_0 * mask_1).sum() / n_basin_1  # V_0 high for basin 1
        loss_contrastive_1 = (contrastive_1 * mask_0).sum() / n_basin_0  # V_1 high for basin 0
        # Apply class weight: contrastive_0 is for basin 1 samples (minority)
        loss_contrastive = self.class_weight * loss_contrastive_0 + loss_contrastive_1

        # 3. Attractor Loss
        # For the last pair of each trajectory, x_tp1 is the terminal state
        # (attractor proxy). Penalize V_i(x_tp1) for basin i to drive V toward
        # zero at the attractor.
        if is_last_pair is not None and self.lambda_attractor > 0:
            final_mask = is_last_pair.unsqueeze(-1)  # (N, 1) bool

            final_and_basin_0 = (final_mask & mask_0.bool()).float()
            n_final_0 = final_and_basin_0.sum() + 1e-8
            loss_attractor_0 = (V_0_tp1 * final_and_basin_0).sum() / n_final_0

            final_and_basin_1 = (final_mask & mask_1.bool()).float()
            n_final_1 = final_and_basin_1.sum() + 1e-8
            loss_attractor_1 = (V_1_tp1 * final_and_basin_1).sum() / n_final_1

            loss_attractor = loss_attractor_0 + self.class_weight * loss_attractor_1
        else:
            loss_attractor = torch.tensor(0.0, device=x_t.device)
            loss_attractor_0 = torch.tensor(0.0, device=x_t.device)
            loss_attractor_1 = torch.tensor(0.0, device=x_t.device)

        # Total loss
        total_loss = (
            self.lambda_decrease * loss_decrease +
            self.lambda_contrastive * loss_contrastive +
            self.lambda_attractor * loss_attractor
        )

        return {
            "total_loss": total_loss,
            "loss_decrease": loss_decrease,
            "loss_decrease_0": loss_decrease_0,
            "loss_decrease_1": loss_decrease_1,
            "loss_contrastive": loss_contrastive,
            "loss_contrastive_0": loss_contrastive_0,
            "loss_contrastive_1": loss_contrastive_1,
            "loss_attractor": loss_attractor,
            "loss_attractor_0": loss_attractor_0,
            "loss_attractor_1": loss_attractor_1,
        }

    def extra_repr(self) -> str:
        return (
            f"epsilon={self.epsilon}, margin={self.margin}, "
            f"lambda_decrease={self.lambda_decrease}, lambda_contrastive={self.lambda_contrastive}, "
            f"lambda_attractor={self.lambda_attractor}, "
            f"class_weight={self.class_weight}, focal_gamma={self.focal_gamma}"
        )
