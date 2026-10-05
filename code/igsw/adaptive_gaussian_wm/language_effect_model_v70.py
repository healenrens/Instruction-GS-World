"""V70 full-text Qwen conditioning and object-local flow/regression training."""
from __future__ import annotations

import torch
from torch import nn

from .continuous_effect_expert_v70 import ContinuousEffectExpertV70, EffectExpertBlockV70
from .vlm_conditioner_v70 import VLMConditionerV70, parameter_counts


class LanguageEffectModelV70(nn.Module):
    """Train only this subsystem; the parent's history encoder remains separate.

    Optional ``conditioner`` injection must implement prepare_inputs(batch),
    forward(vlm_inputs)->{last_hidden_state [B,L,2560], attention_mask [B,L]},
    text_blocks, fsdp_wrap_classes, fsdp_ignored_modules and parameter_inventory.
    It is explicit test injection, never an automatic runtime fallback.

    Train agents wrap the text decoder and effect block classes using FSDP.
    Mixed frozen/trainable parents require use_orig_params=True (FSDP1).
    The frozen visual module is exposed separately as fsdp_ignored_modules.
    Call forward, not encode_condition, for FSDP training. For unsharded eval,
    encode_condition once, then sample repeatedly from the same context.
    """

    def __init__(self, model_path, mode="flow", expert_kwargs=None,
                 visual_tokens=4096, text_tokens=512, *, conditioner=None):
        super().__init__()
        self.mode = {"flow": "flow", "regression": "regression"}[mode]
        self.conditioner = (VLMConditionerV70(model_path, visual_tokens, text_tokens)
                            if conditioner is None else conditioner)
        self.expert = ContinuousEffectExpertV70(**(expert_kwargs or {}))

    @property
    def text_blocks(self):
        return self.conditioner.text_blocks

    @property
    def effect_blocks(self):
        return self.expert.blocks

    @property
    def fsdp_wrap_classes(self):
        return (*self.conditioner.fsdp_wrap_classes, EffectExpertBlockV70)

    @property
    def fsdp_ignored_modules(self):
        return self.conditioner.fsdp_ignored_modules

    def prepare_inputs(self, batch):
        return self.conditioner.prepare_inputs(batch)

    def encode_condition(self, vlm_inputs, history):
        return self.expert.encode_condition(self.conditioner(vlm_inputs), history)

    def forward(self, vlm_inputs, history, target_mean, query_valid, noise=None, tau=None):
        # All trainable conditioning and expert work runs under the root FSDP forward.
        condition = self.encode_condition(vlm_inputs, history)
        valid = query_valid.bool() & condition["query_valid"]
        condition["query_valid"] = valid
        mask = valid[:, :, None, None]
        target_mean = target_mean.detach().float().masked_fill(~mask, 0)
        if self.mode == "flow":
            noise = torch.randn_like(target_mean) if noise is None else noise.float()
            noise = noise.masked_fill(~mask, 0)
            tau = (torch.rand(target_mean.shape[0], device=target_mean.device)
                   if tau is None else tau.to(device=target_mean.device, dtype=torch.float32))
            tau = tau.reshape(-1).expand(target_mean.shape[0])
            u_tau = (1 - tau[:, None, None, None]) * noise + tau[:, None, None, None] * target_mean
            target = target_mean - noise
        else:
            tau = target_mean.new_zeros(target_mean.shape[0])
            u_tau = torch.zeros_like(target_mean)
            target = target_mean
        prediction = self.expert(u_tau, tau, condition)
        error = (prediction.float() - target).masked_fill(~mask, 0).square()
        count = valid.sum().float()
        loss = error.sum() / (count.clamp_min(1) * target.shape[-2] * target.shape[-1])
        metrics = {"loss": loss.detach(), "valid_queries": count.detach(),
                   "tau_mean": tau.mean().detach()}
        if "text_token_counts" in vlm_inputs:
            for name in ("visual_token_counts", "text_token_counts",
                         "visual_budget_overflow", "text_budget_overflow"):
                metrics[name + "_mean"] = vlm_inputs[name].float().mean().detach()
                metrics[name + "_max"] = vlm_inputs[name].float().max().detach()
        return {"loss": loss, "metrics": metrics, "prediction": prediction,
                "target": target, "u_tau": u_tau, "tau": tau}

    @torch.no_grad()
    def sample(self, condition, steps=10, noise=None):
        """Return pre-tanh mean [B,K,4,64]; flow uses Euler tau=0->1."""
        b, k = condition["query_valid"].shape
        state = condition["history_tokens"]
        if self.mode == "regression":
            u = torch.zeros(b, k, 4, 64, device=state.device, dtype=torch.float32)
            return self.expert(u, u.new_zeros(b), condition).float()
        u = (torch.randn(b, k, 4, 64, device=state.device, dtype=torch.float32)
             if noise is None else noise.to(device=state.device, dtype=torch.float32).clone())
        mask = condition["query_valid"][:, :, None, None]
        u = u.masked_fill(~mask, 0)
        dt = 1.0 / steps
        for step in range(steps):
            tau = u.new_full((b,), step * dt)
            u = u + dt * self.expert(u, tau, condition).float()
        return u.masked_fill(~mask, 0)

    def parameter_inventory(self):
        return {"model": parameter_counts(self),
                **self.conditioner.parameter_inventory(), **self.expert.parameter_inventory()}
