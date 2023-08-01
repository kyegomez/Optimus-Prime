import torch
from optimus_prime.attend import Attend

model = Attend(dim=512, dim_head=64, heads=64, q_bucket_size=128, k_bucket_size=128, parallel=False, mixed_precision=False, Flash2=True)
q = torch.randn(1, 8, 512, 64)
k = torch.randn(1, 8, 512, 64)
v = torch.randn(1, 8, 512, 64)
out, _ = model(q, k, v)
assert out.shape == (1, 8, 512, 64)
