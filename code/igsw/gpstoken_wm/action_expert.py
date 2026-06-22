"""π0-style ACTION EXPERT (flow-matching) for the GPSToken world-model VLA (agent.md §VLA).

Turns the frozen-Qwen + 1.6B-DiT 3D-motion world model into a VLA by adding a SECOND head that
shares the DiT trunk: a flow-matching transformer that predicts the Δqpos action chunk.

Architecture (confirmed with the user):
  * The expert is a transformer with the SAME number of layers as the DiT trunk (n_l = 28, the Qwen
    text-layer count). Its width is SMALLER (d_act=768) so it lands ~450-530M params (the trunk is
    1.6B at d=1536); the heavy world-knowledge lives in the shared trunk, the expert just reads it.
  * Input tokens = 50 action steps. Each step token = Linear(noised Δqpos[14]) + learned step-pos
    embedding. (50 = the action-chunk horizon.)
  * At EACH layer i, the 50 action tokens:
      (a) self-attend among themselves (causal-free; the chunk is jointly denoised),
      (b) cross-attend to the DiT trunk's layer-i per-token features  h_i[M, d_trunk]  (KV-cache
          style — the trunk is run ONCE per clip, its per-layer features cached and read here),
      (c) cross-attend to the frozen Qwen-VL conditioning  ctx_i[Q, d_trunk]  (the same per-layer
          aggregated VLM context the trunk's DiTBlocks use),
      (d) cross-attend to the PROPRIOCEPTION state  s[Ns, d]  (the current qpos = anchor[14],
          embedded into Ns conditioning tokens — the robot's "where am I now"). This is a CROSS-
          attention CONDITION, NOT a self-attn sequence member: the state is a given, the 50 action
          tokens read it, and the loss is ONLY on the 50 action tokens (the state is never predicted).
          Conceptually the 51-token (state + 50 action) setup the spec asks for, wired as cross-attn.
      (e) MLP.
    All five sub-blocks are AdaLN-modulated by the flow-matching time t (AdaLN-Zero so a fresh
    expert starts near-identity — but the OUTPUT head is zero-init so v_pred starts at 0, which is
    the safe "no-velocity" start).
  * Output = per-step velocity v_pred[50,14]; flow-matching target v = x1 - x0.

Flow-matching (rectified-flow / π0):
  x1 = normalized Δqpos[50,14];  x0 ~ N(0,I);  t ~ U(0,1);  xt = (1-t)·x0 + t·x1;  target v = x1 - x0.
  loss = MSE(v_pred(xt, t, trunk-KV, vlm-KV), v).  Inference = Euler-integrate the ODE x0 -> x1
  over ~10 steps (sample()).

Action normalization (data/rt2_act/norm_stats.pt):
  arm dims (0-5, 7-12) -> per-dim standardize (x-mean)/std to N(0,1);
  gripper dims (6, 13) near-binary {0,1} -> map to [-1,1] via 2g-1.
  normalize/denormalize round-trip exactly (verified in the smoke test).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dynamics.transformer import SelfAttention, CrossAttention, TimestepEmbed, modulate


# ----------------------------------------------------------------------------- normalization
class ActionNormalizer(nn.Module):
    """Normalize/denormalize the Δqpos action chunk. Stats are BUFFERS (move with the module, saved in
    the ckpt) so train/inference use identical constants. arm dims standardized; gripper dims -> [-1,1]."""

    def __init__(self, mean, std, gripper_dims=(6, 13), dim=14):
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32).clone()
        std = torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6).clone()
        grip = torch.zeros(dim, dtype=torch.bool)
        for g in gripper_dims:
            grip[int(g)] = True
        # For gripper dims we OVERRIDE the standardize stats so the same affine handles both:
        #   normalize(x) = (x - m) / s.  Pick m=0.5, s=0.5 on gripper dims -> (g-0.5)/0.5 = 2g-1 in [-1,1].
        mean = mean.clone(); std = std.clone()
        mean[grip] = 0.5
        std[grip] = 0.5
        self.register_buffer("mean", mean)        # [14]
        self.register_buffer("std", std)          # [14]
        self.register_buffer("grip", grip)        # [14] bool (informational)

    def normalize(self, dq):     # dq [..., 14] raw -> normalized
        return (dq - self.mean.to(dq.device)) / self.std.to(dq.device)

    def denormalize(self, z):    # z [..., 14] normalized -> raw
        return z * self.std.to(z.device) + self.mean.to(z.device)

    @classmethod
    def from_stats_file(cls, path, map_location="cpu", dim=14):
        s = torch.load(path, map_location=map_location, weights_only=False)
        std = s.get("std_safe", s["std"])
        grip = tuple(s.get("gripper_dims", (6, 13)))
        return cls(s["mean"], std, gripper_dims=grip, dim=dim)


# ----------------------------------------------------------------------------- expert block
class ActionExpertBlock(nn.Module):
    """One expert layer: AdaLN(t)-modulated
       [ self-attn | cross-attn(trunk KV) | cross-attn(vlm KV) | cross-attn(STATE KV) | MLP ].

    The 4th sub-block is the PROPRIOCEPTION cross-attn: the 50 action tokens read the current-qpos
    state token(s) as a cross-attention CONDITION (not a self-attn member; loss stays on the 50 action
    tokens only). Kept LAST among the cross-attns so the (sa,ct,cv) modulation-group indices are
    unchanged from the previous 12-group layout.

    AdaLN produces 15 modulation tensors (shift/scale/gate × 5 sub-blocks) from the t-embedding c[B,d].
    Group layout: sa(0-2) ct(3-5) cv(6-8) cs(9-11) mlp(12-14).
    AdaLN-Zero: the gate projection is zero-init so a fresh block is the identity (then the model learns
    the residual). tanh-bounding mirrors the trunk's DiTBlock (prevents the multi-layer blow-up)."""

    def __init__(self, d, n_heads, trunk_dim, vlm_dim, state_dim=None, mlp_ratio=4.0):
        super().__init__()
        state_dim = d if state_dim is None else state_dim
        self.norm_sa = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(d, n_heads)
        self.norm_ct = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.cross_trunk = CrossAttention(d, n_heads, ctx_dim=trunk_dim)   # KV from DiT trunk layer-i feats
        self.norm_cv = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.cross_vlm = CrossAttention(d, n_heads, ctx_dim=vlm_dim)       # KV from frozen Qwen layer-i ctx
        self.norm_cs = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.cross_state = CrossAttention(d, n_heads, ctx_dim=state_dim)   # KV from proprioception state token(s)
        self.norm_mlp = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        hidden = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, d))
        # 15 modulation tensors: (shift,scale,gate) for sa, ct, cv, cs, mlp
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 15 * d))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c, trunk_kv, trunk_mask, vlm_kv, vlm_mask, state_kv, state_mask=None):
        # x [B,A,d] action tokens; c [B,d] t-embedding; trunk_kv [B,M,trunk_dim]; vlm_kv [B,Q,vlm_dim];
        # state_kv [B,Ns,state_dim] proprioception condition (current qpos embedded).
        (sa_sh, sa_sc, sa_g, ct_sh, ct_sc, ct_g, cv_sh, cv_sc, cv_g,
         cs_sh, cs_sc, cs_g, mlp_sh, mlp_sc, mlp_g) = self.ada(c).chunk(15, dim=-1)
        sa_sc, ct_sc, cv_sc, cs_sc, mlp_sc = (torch.tanh(sa_sc), torch.tanh(ct_sc), torch.tanh(cv_sc),
                                              torch.tanh(cs_sc), torch.tanh(mlp_sc))
        sa_g, ct_g, cv_g, cs_g, mlp_g = (torch.tanh(sa_g), torch.tanh(ct_g), torch.tanh(cv_g),
                                         torch.tanh(cs_g), torch.tanh(mlp_g))
        g = (lambda v: v if v.dim() == x.dim() else v.unsqueeze(1))
        x = x + g(sa_g) * self.attn(modulate(self.norm_sa(x), sa_sh, sa_sc))
        x = x + g(ct_g) * self.cross_trunk(modulate(self.norm_ct(x), ct_sh, ct_sc), trunk_kv, trunk_mask)
        x = x + g(cv_g) * self.cross_vlm(modulate(self.norm_cv(x), cv_sh, cv_sc), vlm_kv, vlm_mask)
        x = x + g(cs_g) * self.cross_state(modulate(self.norm_cs(x), cs_sh, cs_sc), state_kv, state_mask)
        x = x + g(mlp_g) * self.mlp(modulate(self.norm_mlp(x), mlp_sh, mlp_sc))
        return x


# ----------------------------------------------------------------------------- expert
class ActionExpert(nn.Module):
    """Flow-matching action expert. n_layers MUST equal the trunk's block count (one KV per layer)."""

    def __init__(self, n_layers, action_dim=14, action_steps=50, d=768, n_heads=12,
                 trunk_dim=1536, vlm_dim=1536, mlp_ratio=4.0, n_state_tokens=1):
        super().__init__()
        self.n_layers, self.action_dim, self.action_steps = n_layers, action_dim, action_steps
        self.d = d
        self.n_state_tokens = int(n_state_tokens)
        # input: noised Δqpos[A,14] -> d, + learned per-step positional embedding
        self.in_proj = nn.Linear(action_dim, d)
        self.step_emb = nn.Parameter(torch.randn(action_steps, d) * 0.02)
        # PROPRIOCEPTION: current qpos (anchor[14]) -> Ns state token(s) of width d. A small MLP embeds the
        # raw qpos; a learned per-state-token positional embedding lets it expand to Ns tokens. The state is
        # consumed ONLY as cross-attn KV in every block (it is a condition, never predicted). The same state
        # KV is reused at all layers (one embedding, layer-shared) — the per-layer AdaLN gate learns how
        # strongly each layer reads it.
        self.state_embed = nn.Sequential(nn.Linear(action_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.state_pos = nn.Parameter(torch.randn(self.n_state_tokens, d) * 0.02)
        # flow-matching time embedding (continuous t in [0,1] -> d) reusing the DiT TimestepEmbed
        self.t_embed = TimestepEmbed(d)
        self.blocks = nn.ModuleList([
            ActionExpertBlock(d, n_heads, trunk_dim, vlm_dim, state_dim=d, mlp_ratio=mlp_ratio)
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(d, eps=1e-6)
        self.out_head = nn.Linear(d, action_dim)
        nn.init.zeros_(self.out_head.weight)     # zero-init -> v_pred starts at 0 (safe no-velocity start)
        nn.init.zeros_(self.out_head.bias)

    def embed_state(self, state):
        """state [B,14] raw current-qpos (anchor) -> state KV [B,Ns,d] (the proprioception condition)."""
        s = self.state_embed(state.to(self.state_pos.dtype))             # [B,d]
        return s[:, None, :] + self.state_pos[None]                      # [B,Ns,d]

    def forward(self, xt, t, trunk_kv, vlm_kv, state_kv, trunk_mask=None, vlm_mask=None):
        """xt [B,A,14] noised action; t [B] in [0,1]; trunk_kv list[n_layers] of [B,M,trunk_dim];
        vlm_kv list[n_layers] of [B,Q,vlm_dim]; state_kv [B,Ns,d] proprioception condition.
        Returns v_pred [B,A,14]."""
        assert len(trunk_kv) == self.n_layers and len(vlm_kv) == self.n_layers, \
            f"need {self.n_layers} KV per source, got trunk={len(trunk_kv)} vlm={len(vlm_kv)}"
        # t-embed scaled to the TimestepEmbed's integer-period range so continuous t in [0,1] is resolved
        c = self.t_embed(t.float() * 1000.0)                              # [B,d]
        x = self.in_proj(xt) + self.step_emb[None]                        # [B,A,d]
        for i, blk in enumerate(self.blocks):
            x = blk(x, c, trunk_kv[i], trunk_mask, vlm_kv[i], vlm_mask, state_kv, None)
        return self.out_head(self.out_norm(x))                           # [B,A,14] velocity

    # ---- flow-matching training objective ----
    def flow_loss(self, x1, trunk_kv, vlm_kv, state_kv, trunk_mask=None, vlm_mask=None, reduce=True):
        """x1 [B,A,14] = NORMALIZED target Δqpos; state_kv [B,Ns,d]. Returns (loss, logs).
        reduce=False returns per-sample loss [B] (used by the batched-vs-per-sample verification)."""
        B = x1.shape[0]
        x0 = torch.randn_like(x1)
        t = torch.rand(B, device=x1.device, dtype=x1.dtype)
        tb = t[:, None, None]
        xt = (1.0 - tb) * x0 + tb * x1
        v_target = x1 - x0
        v_pred = self.forward(xt, t, trunk_kv, vlm_kv, state_kv, trunk_mask, vlm_mask)
        if reduce:
            loss = F.mse_loss(v_pred, v_target)
        else:
            loss = F.mse_loss(v_pred, v_target, reduction="none").mean(dim=(1, 2))   # [B]
        return loss, {"flow": loss.detach(), "v_norm": v_pred.detach().norm(dim=-1).mean()}

    # ---- inference: Euler-integrate the ODE x0 -> x1 ----
    @torch.no_grad()
    def sample(self, trunk_kv, vlm_kv, state_kv, trunk_mask=None, vlm_mask=None,
               n_steps=10, device=None, dtype=None, x0=None):
        """Returns NORMALIZED Δqpos[B,A,14] (caller denormalizes). B inferred from KV.
        x0 (optional): fixed initial noise [B,A,14] — pass it to make sampling deterministic for tests."""
        ref = trunk_kv[0]
        B = ref.shape[0]
        device = device or ref.device
        dtype = dtype or torch.float32
        x = x0.to(device=device, dtype=dtype) if x0 is not None else \
            torch.randn(B, self.action_steps, self.action_dim, device=device, dtype=dtype)
        dt = 1.0 / n_steps
        for s in range(n_steps):
            t = torch.full((B,), s * dt, device=device, dtype=dtype)
            v = self.forward(x, t, trunk_kv, vlm_kv, state_kv, trunk_mask, vlm_mask)
            x = x + dt * v
        return x

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def estimate_params(n_layers=28, action_dim=14, action_steps=50, d=768, mlp_ratio=4.0, n_state_tokens=1):
        """Analytic param count for the FULL config (so we don't have to instantiate the big model)."""
        per = 0
        per += 4 * d * d + 4 * d                                   # self-attn: qkv(3) + proj(1), with bias
        # cross-attn x3 (trunk, vlm, STATE): q(d->d) + kv(ctx->2d) + proj(d->d). Assume ctx_dim ~= d.
        per += 3 * (d * d + d + 2 * d * d + 2 * d + d * d + d)
        hid = int(d * mlp_ratio)
        per += d * hid + hid + hid * d + d                        # mlp
        per += d * (15 * d) + 15 * d                              # adaLN linear (15 groups: sa,ct,cv,cs,mlp)
        core = per * n_layers
        extra = (action_dim * d + d) + action_steps * d           # in_proj + step_emb
        extra += (action_dim * d + d) + (d * d + d) + n_state_tokens * d   # state_embed(2 linears) + state_pos
        extra += d * d + d + d * d + d                            # t_embed mlp (2 linears, dim d)
        extra += (d + d * d) + (d * action_dim + action_dim)      # out_norm(affine) + out_head
        return core + extra
