"""Apply the same GPU-isolation and full-denominator contract to v2."""
import sys
from pathlib import Path
import pytest
import test_spatial_parallel as contract

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_spatial_vsibench_parallel_v2 as parallel_v2


@pytest.fixture(autouse=True)
def use_v2(monkeypatch):
    monkeypatch.setattr(contract, 'parallel', parallel_v2)


test_all_5130_inputs_have_one_worker_and_keep_per_worker_order = contract.test_all_5130_inputs_have_one_worker_and_keep_per_worker_order
test_workers_cannot_share_a_single_allocated_gpu = contract.test_workers_cannot_share_a_single_allocated_gpu
test_each_worker_receives_a_distinct_allocated_device_and_log = contract.test_each_worker_receives_a_distinct_allocated_device_and_log

