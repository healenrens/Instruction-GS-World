"""Deployable RGB-only Object State with training-only relation evidence."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .causal_student_tracklets import CausalStudentTrackletEncoder
from .object_state_target_v52 import ObjectStatePredictions, visible_track_mean, weighted_mean
from .point_track_compositional_decoder import PointTrackCompositionalDecoder
from .point_track_object_state import PointTrackObjectStateEncoder
from .point_track_teacher import sample_patch_field
from .relation_semantic_objective_v54 import relation_semantic_object_state_terms
from .trajectory_relation_teacher_v54 import build_trajectory_relation_teacher_v54


def select_state_batch(state: dict[str, torch.Tensor], indices: torch.Tensor) -> dict:
    return {
        name: value.index_select(0, indices)
        for name, value in state.items()
    }


class RelationSemanticObjectStateModel(nn.Module):
    """The tracker supervises a subset and is never part of the model state."""

    def __init__(self, config):
        super().__init__()
        config.validate()
        self.config = config
        self.student_tracklets = CausalStudentTrackletEncoder(config)
        self.state_encoder = PointTrackObjectStateEncoder(config)
        self.decoder = PointTrackCompositionalDecoder(config)
        self.motion_readout = nn.Linear(
            config.dynamic_dim, len(config.dynamic_horizons) * 2
        )
        self.identity_to_semantic = nn.Sequential(
            nn.LayerNorm(config.identity_dim),
            nn.Linear(config.identity_dim, config.state_dim),
            nn.GELU(),
            nn.Linear(config.state_dim, config.patch_dim),
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
        reconstruction, assignment = self.decoder(
            flat, coordinates.flatten(0, 1), valid.flatten(0, 1)
        )
        return (
            reconstruction.reshape(batch, frames, *reconstruction.shape[1:]),
            assignment.reshape(batch, frames, *assignment.shape[1:]),
        )

    def _track_predictions(self, state, encoder_assignment, decoder_assignment, teacher):
        objects = encoder_assignment[..., : self.config.object_slots]
        object_probability = objects.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        normalized = (objects / object_probability).detach()
        root_identity = F.normalize(state["identity"].float(), dim=-1, eps=1e-6)
        identity = torch.einsum("btpk,btkd->btpd", normalized, root_identity)
        semantic_identity = F.normalize(
            self.identity_to_semantic(identity), dim=-1, eps=1e-6
        )
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
        prediction = ObjectStatePredictions(
            assignment=encoder_assignment,
            identity=identity,
            motion=motion,
            center=center,
            visibility=visibility,
            presence=presence,
            decoder_assignment=decoder_assignment,
        )
        return prediction, semantic_identity

    def forward(
        self,
        patches,
        coordinates,
        valid,
        frame_times,
        point_tracks,
        teacher_indices,
        grid_hw,
    ):
        if point_tracks is None:
            raise ValueError("v54 training requires external point-track evidence")
        if teacher_indices.ndim != 1 or len(teacher_indices) != len(point_tracks.coordinates):
            raise ValueError("v54 teacher indices differ from point-track batch")
        if int(teacher_indices.min()) < 0 or int(teacher_indices.max()) >= len(patches):
            raise ValueError("v54 teacher index is outside the student batch")

        tracklets, state = self.encode_student(patches, coordinates, valid, frame_times)
        reconstruction, decoder_assignment = self.decode_sequence(
            state, coordinates, valid
        )
        target = F.normalize(patches.float(), dim=-1, eps=1e-6)
        reconstruction_loss = weighted_mean(
            1.0 - (reconstruction.float() * target).sum(dim=-1), valid.float()
        )

        teacher_state = select_state_batch(state, teacher_indices)
        teacher_frame_times = frame_times.index_select(0, teacher_indices)
        teacher = build_trajectory_relation_teacher_v54(
            point_tracks, self.config, teacher_frame_times
        )
        sampled_encoder = sample_patch_field(
            teacher_state["assignment"].float(), point_tracks.coordinates, grid_hw
        ).clamp_min(0.0)
        sampled_decoder = sample_patch_field(
            decoder_assignment.index_select(0, teacher_indices).float(),
            point_tracks.coordinates,
            grid_hw,
        ).clamp_min(0.0)
        prediction, semantic_identity = self._track_predictions(
            teacher_state, sampled_encoder, sampled_decoder, teacher
        )
        terms = relation_semantic_object_state_terms(
            prediction,
            semantic_identity,
            teacher,
            point_tracks,
            self.config,
        )
        loss = terms["target_total"] + self.config.reconstruction_weight * reconstruction_loss
        object_pair = (
            teacher.object_confidence[:, :, None]
            * teacher.object_confidence[:, None]
        )
        parts = {
            "loss": loss,
            "loss_object_state": loss,
            "loss_external_target": terms["target_total"],
            "loss_reconstruction_auxiliary": reconstruction_loss,
            "teacher_batch_fraction": patches.new_tensor(
                len(teacher_indices) / len(patches), dtype=torch.float32
            ),
            "teacher_same_relation_weight": teacher.same_confidence.mean(),
            "teacher_different_relation_weight": teacher.different_confidence.mean(),
            "teacher_supervised_same_relation_weight": (
                teacher.same_confidence * object_pair
            ).mean(),
            "teacher_supervised_different_relation_weight": (
                teacher.different_confidence * object_pair
            ).mean(),
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
            raise RuntimeError("v54 Object State loss is non-finite")
        return {
            "loss": loss,
            "parts": parts,
            "state": state,
            "student_tracklets": tracklets,
            "teacher": teacher,
            "teacher_indices": teacher_indices,
            "prediction": prediction,
            "semantic_identity": semantic_identity,
            "reconstruction": reconstruction,
            "decoder_assignment": decoder_assignment,
        }
