from types import SimpleNamespace

import pandas as pd
import pytest

from mile.constants import CARLA_FPS
from mile.data.dataset import CarlaDataset


@pytest.mark.parametrize('length,sequence_length,stride,expected', [
    (1, 1, 1, [[0]]),
    (3, 3, 1, [[0, 1, 2]]),
    (5, 3, 2, [[0, 2, 4]]),
    (6, 3, 2, [[0, 2, 4], [1, 3, 5]]),
    (2, 3, 1, []),
])
def test_every_sequence_that_fits_in_the_run_is_available(length, sequence_length, stride, expected):
    dataset = CarlaDataset.__new__(CarlaDataset)
    dataset.cfg = SimpleNamespace(
        DATASET=SimpleNamespace(
            FILTER_NORM_REWARD=0.0,
            STRIDE_SEC=stride / CARLA_FPS,
            FILTER_BEGINNING_OF_RUN_SEC=0.0,
        ),
        EVAL=SimpleNamespace(DATASET_REDUCTION=False),
    )
    dataset.sequence_length = sequence_length
    dataset.dataset_path = 'synthetic'
    dataset.data = {'Town01/0000': pd.DataFrame({'reward': [1.0] * length})}
    actual = [indices for _, indices in dataset.get_data_pointers()]
    assert actual == expected
