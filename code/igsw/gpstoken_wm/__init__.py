"""GPSToken-JEPA sparse 3D-Gaussian world model (PLAN_GPSTOKEN_JEPA_zh.md, agent.md §93-95)."""
from .wm_model import GPSTokenWM
from .sigreg import SIGReg
from .tokens import place_tokens, sample_grid_feat, project_to_uv, to_cam, cam_to_world
from .action_expert import ActionExpert, ActionExpertBlock, ActionNormalizer
from . import losses

__all__ = ["GPSTokenWM", "SIGReg", "place_tokens", "sample_grid_feat", "project_to_uv",
           "to_cam", "cam_to_world", "losses",
           "ActionExpert", "ActionExpertBlock", "ActionNormalizer"]
