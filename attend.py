import pytest
import torch
from optimus_prime.attend import Attend


def test_forward_pass():
    model = Attend(dim=512, dim_head=64, q_bucket_size=128, k_bucket_size=128, parallel=False, mixed_precision=False, flash=True)
    q = torch.randn(1, 8, 512, 64)
    k = torch.randn(1, 8, 512, 64)
    v = torch.randn(1, 8, 512, 64)
    out, _ = model(q, k, v)
    assert out.shape == (1, 8, 512, 64)

def test_backward_pass():
    model = Attend(dim=512, dim_head=64, q_bucket_size=128, k_bucket_size=128, parallel=False, mixed_precision=False, flash=True)
    q = torch.randn(1, 8, 512, 64, requires_grad=True)
    k = torch.randn(1, 8, 512, 64, requires_grad=True)
    v = torch.randn(1, 8, 512, 64, requires_grad=True)
    out, _ = model(q, k, v)
    out.sum().backward()
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None

def test_memory_usage():
    model = Attend(dim=512, dim_head=64, q_bucket_size=128, k_bucket_size=128, parallel=False, mixed_precision=False, flash=True)
    q = torch.randn(1, 8, 512, 64)
    k = torch.randn(1, 8, 512, 64)
    v = torch.randn(1, 8, 512, 64)
    before_memory = torch.cuda.memory_allocated()
    out, _ = model(q, k, v)
    after_memory = torch.cuda.memory_allocated()
    assert after_memory - before_memory < 1e6  # less than 1MB increase

def test_execution_time():
    import time
    model = Attend(dim=512, dim_head=64, q_bucket_size=128, k_bucket_size=128, parallel=False, mixed_precision=False, flash=True)
    q = torch.randn(1, 8, 512, 64)
    k = torch.randn(1, 8, 512, 64)
    v = torch.randn(1, 8, 512, 64)
    start_time = time.time()
    out, _ = model(q, k, v)
    end_time = time.time()
    assert end_time - start_time < 1  # less than 1 second

def test_error_rate():
    model = Attend(dim=512, dim_head=64, q_bucket_size=128, k_bucket_size=128, parallel=False, mixed_precision=False, flash=True)
    q = torch.randn(1, 8, 512, 64)
    k = torch.randn(1, 8, 512, 64)
    v = torch.randn(1, 8, 512, 64)
    out, _ = model(q, k, v)
    assert (out != out).sum() == 0  # no NaN values