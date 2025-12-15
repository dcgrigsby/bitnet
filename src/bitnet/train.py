from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from bitnet.config import BitNetConfig
from bitnet.transformer import BitNetModel


class TwoStageLRScheduler:
    """Two-stage learning rate scheduler for BitNet b1.58

    Based on "The Era of 1-bit LLMs: Training Tips, Code and FAQ" (Table 2):
      - Stage 1 (0-50%): Warmup then linear decay from peak_lr -> stage2_lr
      - Stage 2 (50-100%): Linear decay from stage2_lr -> final_lr (~0)

    The stage 2 starting LR is 2/3 of peak LR across all model sizes.
    The final LR decays to 1% of peak LR.

    Args:
        optimizer: PyTorch optimizer
        config: BitNetConfig with .learning_rate and .warmup_steps
        total_steps: Total optimizer steps for entire training run"""

    def __init__(
        self, optimizer: optim.Optimizer, config: BitNetConfig, total_steps: int
    ) -> None:
        self.optimizer: optim.Optimizer = optimizer
        self.total_steps: int = total_steps
        self.current_step: int = 1

        self.warmup_steps: int = config.warmup_steps
        self.peak_lr: float = config.learning_rate

        self.stage_boundary: int = total_steps // 2

        self.stage2_lr: float = self.peak_lr * 2.0 / 3.0
        self.final_lr: float = self.peak_lr * 0.001

    def _lr_at_step(self, step: int) -> float:
        """Compute learning rate for a given global step (0-based)."""

        # Warmup: 0 -> peak_lr
        if step < self.warmup_steps:
            return self.peak_lr * (step / self.warmup_steps)

        # Stage 1: peak_lr -> stage2_lr (linear decay)
        if step < self.stage_boundary:
            progress = (step - self.warmup_steps) / (
                self.stage_boundary - self.warmup_steps
            )
            return self.peak_lr + (self.stage2_lr - self.peak_lr) * progress

        # Stage 2: stage2_lr -> final_lr (linear decay)
        progress = (step - self.stage_boundary) / (
            self.total_steps - self.stage_boundary
        )
        return self.stage2_lr + (self.final_lr - self.stage2_lr) * progress

    def step(self) -> None:
        """Advance one step and update optimizer learning rates."""

        lr = self._lr_at_step(self.current_step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        self.current_step += 1

    def get_lr(self) -> float:
        """Get current learning rate without advancing."""
        return self._lr_at_step(self.current_step)


class TwoStageWDScheduler:
    """Two-stage weight decay scheduler for BitNet b1.58

    Based on "The Era of 1-bit LLMs: Training Tips, Code and FAQ":
    - Stage 1 (0-50%): WD = 0.1 (regularization)
    - Stage 2 (50-100%): WD = 0.05 (moderate regularization to prevent mode collapse)

    In 1-bit training, latent weight magnitude acts as confidence score.
    Setting WD=0.0 in Stage 2 can cause mode collapse on small models/datasets.
    We use WD=0.05 as a middle ground for stability.
    """

    def __init__(self, optimizer: optim.Optimizer, total_steps: int) -> None:
        self.optimizer: optim.Optimizer = optimizer
        self.total_steps: int = total_steps
        self.current_step: int = 0
        self.stage1_wd: float = 0.1
        self.stage2_wd: float = 0.05
        self.stage1_steps: int = total_steps // 2

    def step(self) -> None:
        """Update weight decay based on current step"""

        wd = self.stage1_wd if self.current_step < self.stage1_steps else self.stage2_wd

        for param_group in self.optimizer.param_groups:
            param_group["weight_decay"] = wd

        self.current_step += 1


def train_step(
    model: BitNetModel,
    batch: torch.Tensor,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    lr_scheduler: Optional[TwoStageLRScheduler] = None,
    wd_scheduler: Optional[TwoStageWDScheduler] = None,
) -> float:
    """Single training step with scheduler updates.

    Args:
        model: BitNetModel instance
        batch: Batch of token IDs of shape [batch_size, seq_len]
        optimizer: Optimizer instance
        loss_fn: Loss function
        lr_scheduler: Learning rate scheduler (optional)
        wd_scheduler: Weight decay scheduler (optional)

    Returns:
        Loss value
    """

    optimizer.zero_grad()

    # Forward pass
    logits = model(batch)

    # Compute loss (next token prediction)
    # Shift targets: predict next token given current tokens
    _ = batch.shape  # Shape available if needed
    flat_logits = logits[:, :-1, :].reshape(-1, model.config.vocab_size)
    flat_targets = batch[:, 1:].reshape(-1)

    loss = loss_fn(flat_logits, flat_targets)

    # Backward pass
    loss.backward()

    # Gradient clipping (important for stability)
    _ = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Optimizer step
    optimizer.step()

    # Update schedulers
    if lr_scheduler is not None:
        lr_scheduler.step()
    if wd_scheduler is not None:
        wd_scheduler.step()

    return loss.item()


def train_epoch(
    model: BitNetModel,
    dataloader: DataLoader[Any],
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    lr_scheduler: Optional[TwoStageLRScheduler] = None,
    wd_scheduler: Optional[TwoStageWDScheduler] = None,
) -> float:
    """Train for one epoch.

    Args:
        model: BitNetModel instance
        dataloader: DataLoader
        optimizer: Optimizer
        loss_fn: Loss function
        lr_scheduler: Learning rate scheduler (optional)
        wd_scheduler: Weight decay scheduler (optional)

    Returns:
        Average loss over epoch
    """

    total_loss = 0.0
    num_batches = 0

    _ = model.train()

    for batch in dataloader:
        batch = batch[0]  # TensorDataset returns tuple
        loss = train_step(model, batch, optimizer, loss_fn, lr_scheduler, wd_scheduler)
        total_loss += loss
        num_batches += 1

    return total_loss / num_batches
