import pytest
import torch

from core.training_utils import setup_distributed


@pytest.mark.parametrize("current", [0, 2])
def test_single_gpu_has_explicit_current_index(monkeypatch, current):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current)
    distributed, rank, world_size, local_rank, device = setup_distributed()
    assert (distributed, rank, world_size, local_rank) == (False, 0, 1, 0)
    assert device == torch.device("cuda", current)
    # Reproduce the cluster's stricter mem_get_info device requirement without
    # needing CUDA hardware in the local test environment.
    def strict_memory_query(device):
        if device.index is None:
            raise ValueError("Expected a device with a specified index")
        assert device.index == current
        return 8, 16
    monkeypatch.setattr(torch.cuda, "mem_get_info", strict_memory_query)
    assert torch.cuda.mem_get_info(device) == (8, 16)


def test_cpu_setup_does_not_query_cuda(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: pytest.fail("CPU startup queried CUDA"))
    assert setup_distributed()[-1] == torch.device("cpu")


def test_ddp_uses_local_rank(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    selected = []
    initialized = []
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    monkeypatch.setattr(torch.distributed, "init_process_group", lambda **kwargs: initialized.append(kwargs))
    assert setup_distributed() == (True, 2, 4, 2, torch.device("cuda", 2))
    assert selected == [2]
    assert initialized == [{"backend": "nccl", "init_method": "env://"}]
