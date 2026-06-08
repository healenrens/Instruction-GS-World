"""CPU unit test for v7-min step2 feature plumbing (relevance as input token).
Tests: GaussianTokenizer(feature_dim=1) + SCGSRollout carrying control features through rollout.
No Qwen/GPU needed."""
import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from igsw.gaussians.types import GaussianSet
from igsw.dynamics.tokenizer import GaussianTokenizer
from igsw.dynamics.scgs import SCGSRollout

torch.manual_seed(0)
N = 500
g0 = GaussianSet(
    means=torch.randn(N, 3), quats=torch.tensor([1., 0, 0, 0]).repeat(N, 1),
    scales=torch.rand(N, 3) * 0.01 + 0.001, opacities=torch.rand(N) * 0.5 + 0.25,
    colors=torch.rand(N, 3), features=torch.rand(N, 1))   # relevance feature [N,1]

# 1) tokenizer consumes the feature channel
tok = GaussianTokenizer(d_model=64, num_freqs=6, feature_dim=1)
print("tokenizer in_dim:", tok.in_dim, "(includes +1 for relevance)")
x = tok.tokenize_tensors(g0.means[None], g0.quats[None], g0.log_scales[None],
                         g0.opacity_logits[None], g0.colors[None], g0.features[None])
assert x.shape == (1, N, 64), x.shape
print("tokenize w/ feature OK ->", tuple(x.shape))

# 2) SCGS carries control features + preserves through rollout
roll = SCGSRollout(g0, n_control=32, k=4)
assert roll.control0.features is not None and roll.control0.features.shape == (32, 1)
print("control0 carries features:", tuple(roll.control0.features.shape))

def dummy_delta_fn(state, step_idx):
    M = state.means.shape[1]
    # verify the rollout actually feeds features into the delta fn
    assert state.features is not None and state.features.shape == (1, M, 1), state.features
    z3 = torch.zeros(1, M, 3); z1 = torch.zeros(1, M, 1)
    return z3, z3, z3, z1, z3, None

dense_states, deltas, ctrl_traj = roll.rollout(dummy_delta_fn, K=3)
assert len(dense_states) == 3
print("rollout OK, control features fed every step; dense states:", len(dense_states))
print("[OK] v7-min step2 feature plumbing correct")
