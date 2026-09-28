"""Complete object-video-sequence configuration, independent of legacy slot roles."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ObjectVideoConfigV69:
    checkpoint_version: int = 69
    architecture: str = "pretrained_query_object_video_sequence_v2"
    encoder: str = "dinov3_vitl16"
    perception_dim: int = 1024
    patch_size: int = 16
    width: int = 512
    heads: int = 8
    object_queries: int = 16
    local_carriers: int = 8
    observation_layers: int = 4
    memory_layers: int = 4
    posterior_layers: int = 4
    posterior_geometry: bool = True
    dynamics_layers: int = 8
    effect_tokens: int = 4
    effect_dim: int = 64
    readout_width: int = 512
    history_frames: int = 16
    future_frames: int = 25
    history_seconds: float = 3.0
    future_seconds: float = 5.0
    history_min_seconds: float = 3.0
    future_min_seconds: float = 5.0
    measurement_points: int = 256
    ema: float = 0.996
    feature_weight: float = 1.0
    binding_weight: float = 1.0
    correspondence_weight: float = 1.0
    trajectory_weight: float = 1.0
    relative_motion_weight: float = 0.25
    latent_weight: float = 0.25
    observation_weight: float = 0.1
    path_weight: float = 0.25
    effect_rate_weight: float = 0.001
    source_weights: str = "episode_uniform"

    def to_dict(self):
        return asdict(self)

    @property
    def tokens_per_object(self):
        return 1 + self.local_carriers


def parameter_inventory(modules):
    return {name: {"parameters": sum(p.numel() for p in module.parameters()),
                   "trainable_parameters": sum(p.numel() for p in module.parameters() if p.requires_grad)}
            for name, module in modules.items()}
