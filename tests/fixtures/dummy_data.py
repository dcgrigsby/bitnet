from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from bitnet.config import BitNetConfig


def create_dummy_dataloader(
    config: BitNetConfig, num_batches: int = 4, batch_size: int = 2, seq_len: int = 10
) -> DataLoader[Any]:
    """Create a dummy dataloader for testing.

    Args:
        config: Model config
        num_batches: Number of batches
        batch_size: Batch size
        seq_len: Sequence length

    Returns:
        DataLoader with dummy data
    """
    data = torch.randint(0, config.vocab_size, (num_batches * batch_size, seq_len))
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader
