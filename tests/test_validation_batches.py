from types import SimpleNamespace

import torch

from mile.data.dataset import DataModule


def test_validation_retains_the_last_partial_batch():
    config = SimpleNamespace(BATCHSIZE=2, RECEPTIVE_FIELD=1, FUTURE_HORIZON=1, N_WORKERS=0)
    module = DataModule(config)
    module.val_dataset = torch.utils.data.TensorDataset(torch.arange(3))
    module.val_sampler = None
    batches = list(module.val_dataloader())
    assert torch.cat([batch[0] for batch in batches]).tolist() == [0, 1, 2]


def test_validation_smaller_than_one_batch_is_not_empty():
    config = SimpleNamespace(BATCHSIZE=4, RECEPTIVE_FIELD=1, FUTURE_HORIZON=1, N_WORKERS=0)
    module = DataModule(config)
    module.val_dataset = torch.utils.data.TensorDataset(torch.arange(1))
    module.val_sampler = None
    assert len(list(module.val_dataloader())) == 1
