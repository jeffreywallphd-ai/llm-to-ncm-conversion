import pytest
import torch
from ncm_lab.models.fixtures import tiny_pair


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(7)


@pytest.fixture
def pair():
    return tiny_pair()
