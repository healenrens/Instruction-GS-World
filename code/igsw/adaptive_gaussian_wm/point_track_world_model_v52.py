"""RGB-only Object State model supervised by trajectory relations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .causal_student_tracklets import CausalStudentTrackletEncoder
from .object_state_target_v52 import (
    ObjectStatePredictions,
    object_state_target_terms,
    visible_track_mean,
    weighted_mean,
)
from .point_track_compositional_decoder import PointTrackCompositionalDecoder
from .point_track_object_state import PointTrackObjectStateEncoder
from .point_track_teacher import sample_patch_field
from .trajectory_relation_teacher import build_trajectory_relation_teacher


class LearningObjectiveObjectWorldModel(nn.Module):
    """The frozen tracker supplies loss evidence and is absent from deployment."""

    def __init__(self, config):
        super().__init__()
        config.validate()
        self.config = config
        self.student_tracklets = CausalStudentTrackletEncoder(config)
        self.state_encoder = PointTrackObjectStateEncoder(config)
        self.decoder = PointTrackCompositionalDecoder(config)
        self.identity_readout = nn.Linear(config.identity_dim, config.patch_dim)
        self.motion_readout = nn.Linear(
            config.dynamic_dim, len(config.dynamic_horizons) * 2
        )

    def encode_student(self, patches, coordinates, valid, frame_times):
        tracklets = self.student_tracklets(patches, coordinates, valid)
        state = self.state_encoder(
            tracklets.features,
            coordinates,
            valid,
            frame_times,
            tracklets.temporal_residual,
            tracklets.residual_flow,
            tracklets.confidence,
        )
        return tracklets, state

    def decode_sequence(self, state, coordinates, valid):
        batch, frames = state["identity"].shape[:2]
        flat = {
            name: state[name].flatten(0, 1)
            for name in (
                "identity", "dynamic", "center", "log_scale", "support_shape",
                "presence", "visibility", "scene", "transient",
            )
        }
        reconstructed, assignment = self.decoder(
            flat, coordinates.flatten(0, 1), valid.flatten(0, 1)
        )
        return (
            reconstructed.reshape(batch, frames, *reconstructed.shape[1:]),
            assignment.reshape(batch, frames, *assignment.shape[1:]),
        )

    def _track_predictions(self, state, encoder_assignment, decoder_assignment, teacher):
        objects = encoder_assignment[..., : self.config.object_slots]
        object_probability = objects.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        normalized = (objects / object_probability).detach()
        root_identity = F.normalize(
            self.identity_readout(state["identity"].float()), dim=-1, eps=1e-6
        )
        identity = torch.einsum("btpk,btkd->btpd", normalized, root_identity)
        root_motion = self.motion_readout(state["dynamic"].float()).reshape(
            *state["dynamic"].shape[:3], len(self.config.dynamic_horizons), 2
        )
        motion = torch.einsum("btpk,btkhd->btphd", normalized, root_motion)
        center = torch.einsum("btpk,btkd->btpd", normalized, state["center"].float())
        reference = visible_track_mean(objects, teacher.visibility).detach()
        reference = reference / reference.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        visibility = torch.einsum(
            "bpk,btk->btp", reference, state["visibility"].float()
        )
        presence = torch.einsum(
            "bpk,btk->btp", reference, state["presence"].float()
        )
        return ObjectStatePredictions(
            assignment=encoder_assignment,
            identity=identity,
            motion=motion,
            center=center,
            visibility=visibility,
            presence=presence,
            decoder_assignment=decoder_assignment,
        )

    def forward(self, patches, coordinates, valid, frame_times, point_tracks, grid_hw):
        if point_tracks is None:
            raise ValueError("v52 Object State training requires point-track evidence")
        tracklets, state = self.encode_student(patches, coordinates, valid, frame_times)
        teacher = build_trajectory_relation_teacher(point_tracks, self.config, frame_times)
        reconstruction, decoder_assignment = self.decode_sequence(state, coordinates, valid)
        sampled_encoder = sample_patch_field(
            state["assignment"].float(), point_tracks.coordinates, grid_hw
        ).clamp_min(0.0)
        sampled_decoder = sample_patch_field(
            decoder_assignment.float(), point_tracks.coordinates, grid_hw
        ).clamp_min(0.0)
        prediction = self._track_predictions(
            state, sampled_encoder, sampled_decoder, teacher
        )
        terms = object_state_target_terms(
            prediction, teacher, point_tracks, self.config
        )
        normalized_target = F.normalize(patches.float(), dim=-1, eps=1e-6)
        reconstruction_loss = weighted_mean(
            1.0 - (reconstruction.float() * normalized_target).sum(dim=-1),
            valid.float(),
        )
        loss = terms["target_total"] + self.config.reconstruction_weight * reconstruction_loss
        parts = {
            "loss": loss,
            "loss_object_state": loss,
            "loss_target_contract": terms["target_total"],
            "loss_reconstruction_auxiliary": reconstruction_loss,
            "teacher_same_relation_weight": teacher.same_confidence.mean(),
            "teacher_different_relation_weight": teacher.different_confidence.mean(),
            "teacher_object_confidence": teacher.object_confidence.mean(),
            "teacher_scene_confidence": teacher.scene_confidence.mean(),
            "teacher_transient_confidence": teacher.transient_confidence.mean(),
            "teacher_lifecycle_known_fraction": teacher.lifecycle_known.float().mean(),
            "teacher_occluded_fraction": (
                (teacher.lifecycle_state == 1) & teacher.lifecycle_known
            ).float().mean(),
        }
        parts.update({f"target_{name}": value for name, value in terms.items()})
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("v52 Object State loss is non-finite")
        return {
            "loss": loss,
            "parts": parts,
            "state": state,
            "student_tracklets": tracklets,
            "teacher": teacher,
            "prediction": prediction,
            "reconstruction": reconstruction,
            "decoder_assignment": decoder_assignment,
        }
