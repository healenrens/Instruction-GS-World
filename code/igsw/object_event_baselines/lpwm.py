"""Bridge to the official LPWM model; no V71 architecture or loss is substituted."""

from dataclasses import dataclass
import importlib
import inspect
import json
from pathlib import Path
import sys

import torch
from torch import nn

from igsw.object_event_wm.codec import letterbox_video
from . import LPWM_REVISION


CONFIG_ALIASES = {"cdim": "ch", "n_static_frames": "num_static_frames",
                  "ctx_dist": "context_dist", "img_goal_condition": "image_goal_condition"}


@dataclass
class LPWMConfig:
    native: dict
    upstream_revision: str = LPWM_REVISION
    image_policy: str = "letterbox_full_field_of_view"


def load_upstream(repository):
    sys.path.insert(0, str(Path(repository).resolve()))
    return importlib.import_module("models"), importlib.import_module("utils.loss_functions")


def make_config(assets, sequence, text_metadata):
    native = json.loads(Path(assets["native_config"]).read_text())
    original = dict(native)
    native["timestep_horizon"] = sequence["history_frames"]+sequence["future_frames"]-1
    native["cond_steps"] = sequence["history_frames"]
    native["language_max_len"] = text_metadata["sequence_length"]
    native["language_embed_dim"] = text_metadata["width"]
    differences = {key: {"upstream": original.get(key), "effective": value}
                   for key, value in native.items() if original.get(key) != value}
    return LPWMConfig(native, assets["upstream_revision"]), differences


class LPWMVideoModel(nn.Module):
    def __init__(self, config, repository):
        super().__init__()
        self.config = config
        upstream, _ = load_upstream(repository)
        parameters = inspect.signature(upstream.DLP).parameters
        arguments = {name: config.native[CONFIG_ALIASES.get(name, name)]
                     for name in parameters if CONFIG_ALIASES.get(name, name) in config.native}
        self.native = upstream.DLP(**arguments)

    def inventory(self):
        return {"parameters": sum(p.numel() for p in self.parameters()),
                "encoder_particles": self.native.n_kp_enc,
                "decoder_particles": self.native.n_kp_dec,
                "context_dimension": self.native.context_dim,
                "image_size": self.native.image_size,
                "upstream_revision": self.config.upstream_revision,
                "image_policy": self.config.image_policy}

    def images(self, batch):
        return letterbox_video(batch, self.config.native["image_size"])[0]

    def forward(self, batch, reconstruction_loss, warmup=False):
        settings = self.config.native
        images = self.images(batch)
        output = self.native(images, with_loss=True, warmup=warmup,
            lang_embed=batch["text_tokens"].float(), actions=None, actions_mask=None, x_goal=None,
            done_mask=batch["frame_valid"].int(), recon_loss_func=reconstruction_loss,
            **{name: settings[name] for name in ("beta_kl", "beta_dyn", "beta_rec", "kl_balance",
                                                "recon_loss_type", "beta_dyn_rec", "beta_obj")})
        return {"loss": output["loss_dict"]["loss"], "parts": output["loss_dict"]}

    @torch.no_grad()
    def predict_history(self, observed, future_frames, language, seed=17, deterministic=False):
        images = self.images(observed)
        devices = [images.device.index] if images.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if images.is_cuda:
                with torch.cuda.device(images.device):
                    torch.cuda.manual_seed(seed)
            result = self.native.sample_from_x(images, num_steps=future_frames,
                cond_steps=images.shape[1], deterministic=deterministic, use_all_ctx=False,
                n_pred_eq_gt=False, lang_embed=language, actions=None, x_goal=None)
        return result[:, images.shape[1]:]

    @torch.no_grad()
    def reconstruct_posterior(self, batch, language):
        images = self.images(batch)
        th = batch["history_frames"]
        return self.native.sample_from_x(images, num_steps=images.shape[1]-th,
            cond_steps=th, deterministic=True, use_all_ctx=True, n_pred_eq_gt=True,
            lang_embed=language, actions=None, x_goal=None)[:, th:]


def local_reconstruction_loss(assets, normalized_rgb):
    """Use upstream forward functions, replacing only their network-enabled constructors."""
    from torchvision import models

    _, upstream = load_upstream(assets["repository"])

    class LocalVGG(upstream.vgg16):
        def __init__(self):
            nn.Module.__init__(self)
            network = models.vgg16(weights=None, init_weights=False)
            network.load_state_dict(torch.load(assets["vgg16_weights"], map_location="cpu", weights_only=True))
            boundaries = (0, 4, 9, 16, 23, 30)
            self.N_slices = 5
            for index, (start, stop) in enumerate(zip(boundaries, boundaries[1:]), 1):
                setattr(self, f"slice{index}", network.features[start:stop])
            self.requires_grad_(False)

    class LocalLPIPS(upstream.LPIPS):
        def __init__(self):
            nn.Module.__init__(self)
            self.scaling_layer = upstream.ScalingLayer()
            self.chns = [64, 128, 256, 512, 512]
            self.net = LocalVGG()
            for index, channels in enumerate(self.chns):
                setattr(self, f"lin{index}", upstream.NetLinLayer(channels, use_dropout=True))
            weights = torch.load(assets["lpips_weights"], map_location="cpu", weights_only=True)
            heads = nn.ModuleDict({f"lin{index}": getattr(self, f"lin{index}") for index in range(5)})
            heads.load_state_dict({name: value for name, value in weights.items() if name.startswith("lin")})
            self.requires_grad_(False)

    class LocalLoss(upstream.LossLPIPS):
        def __init__(self):
            nn.Module.__init__(self)
            self.pixel_weight = 1.0
            self.perceptual_loss = LocalLPIPS().eval()
            self.perceptual_weight = 0.1
            self.normalized_rgb = normalized_rgb

    return LocalLoss().eval()
