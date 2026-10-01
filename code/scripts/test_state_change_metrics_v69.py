#!/usr/bin/env python3
"""CPU mathematical regression: real change, static controls and reference offsets."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn

from igsw.adaptive_gaussian_wm.query_object_video_encoder_v69 import ObjectVideoStateV69
from igsw.adaptive_gaussian_wm.state_change_evaluation_v69 import (
    aggregate_case_rows_v69, intervene_states_v69, load_state_modules_v69,
    select_held_episodes_v69, state_change_values_v69, summarize_state_case_v69,
)


class StateChangeContract(unittest.TestCase):
    def test_change_evidence_and_controls(self):
        target = torch.zeros(1, 4, 2, 2)
        target[0, 2:, 0, 0] = torch.tensor([.2, .6])  # 10px and 30px on a 101px image.
        target[0, 2:, 1, 1] = .4
        valid = torch.ones(1, 4, 2, dtype=torch.bool)
        valid[0, 1, 1] = False  # Position can be scored; current-to-target change cannot.
        batch = {"native_hw": torch.tensor([[101, 101]]), "case_id": ["clip_a"], "source": ["source_a"],
                 "times": torch.tensor([[-1., 0., 1., 2.]]),
                 "teacher": {"xy": target, "valid": valid, "point_present": torch.ones(1, 2, dtype=torch.bool),
                             "transport_weight": torch.tensor([[1., 0.]])}}
        predictions = {"observed_state": target.clone(), "frozen_state": torch.zeros_like(target),
                       "shuffled_state": target[:, [0, 1, 3, 2]], "frozen_tokens": target.clone(),
                       "frozen_centers": torch.zeros_like(target), "reference_copy": torch.zeros_like(target)}
        values, position_mask, change_mask, moving = state_change_values_v69(predictions, batch, 2, 5.)
        self.assertEqual(int(position_mask.sum()), 4)
        self.assertEqual(int(change_mask.sum()), 2)
        self.assertEqual(int(moving.sum()), 2)
        torch.testing.assert_close(values["observed_state"]["change_error_px"], torch.zeros_like(values["observed_state"]["change_error_px"]))
        torch.testing.assert_close(values["frozen_state"]["normalized_change_error"][moving], torch.ones(2))
        torch.testing.assert_close(values["frozen_state"]["direction_cosine_error"][moving], torch.ones(2))
        torch.testing.assert_close(values["shuffled_state"]["change_error_px"][moving], torch.tensor([20., 20.]))
        shifted = {"observed_state": target + .1}
        shifted_values, _, _, _ = state_change_values_v69(shifted, batch, 2, 5.)
        torch.testing.assert_close(shifted_values["observed_state"]["change_error_px"], torch.zeros_like(values["observed_state"]["change_error_px"]), atol=1e-5, rtol=0)
        self.assertGreater(float(shifted_values["observed_state"]["position_error_px"].mean()), 7.)
        rows, paired = summarize_state_case_v69(predictions, batch, SimpleNamespace(history_frames=2, future_seconds=2.), 5.)
        control = next(row for row in paired if row["condition"] == "frozen_state" and row["pool"] == "all_points")
        self.assertEqual(control["change_error_px_penalty_count"], 2)
        self.assertAlmostEqual(control["change_error_px_penalty_mean"], 20.)
        tiny = next(row for row in rows if row["slice"] == "under_1px" and row["pool"] == "all_points")
        self.assertIsNone(tiny["position_error_px_mean"])
        summary = aggregate_case_rows_v69(rows)
        measured = next(row for row in summary if row["source"] == "source_a" and row["condition"] == "observed_state"
                        and row["pool"] == "all_points" and row["scope"] == "motion" and row["slice"] == "motion_active")
        self.assertEqual(measured["change_error_px_mean"]["count"], 1)
        states = [ObjectVideoStateV69(torch.full((1, 1, 2, 3), float(i)), torch.full((1, 1, 2, 2), float(i)),
                                     torch.tensor([float(i)]), torch.ones(1, 1, dtype=torch.bool)) for i in range(5)]
        for condition in ("frozen_state", "frozen_tokens", "frozen_centers", "shuffled_state"):
            changed = intervene_states_v69(states, 2, condition, 17)
            self.assertIs(changed[0], states[0])
            self.assertIs(changed[1], states[1])
            for frame in range(2, 5):
                torch.testing.assert_close(changed[frame].time, states[frame].time)
                if condition in ("frozen_state", "frozen_tokens"):
                    torch.testing.assert_close(changed[frame].tokens, states[1].tokens)
                if condition in ("frozen_state", "frozen_centers"):
                    torch.testing.assert_close(changed[frame].centers, states[1].centers)
                if condition == "shuffled_state":
                    self.assertFalse(torch.equal(changed[frame].tokens, states[frame].tokens))
        modules = SimpleNamespace(encoder=nn.Linear(2, 2), target_encoder=nn.Linear(2, 2), readout=nn.Linear(2, 2))
        weights = {f"{name}.{key}": value for name in ("encoder", "target_encoder", "readout")
                   for key, value in getattr(modules, name).state_dict().items()}
        weights["posterior.obsolete"] = torch.tensor(0.)
        self.assertEqual(load_state_modules_v69(modules, weights), ["encoder", "target_encoder", "readout"])
        dataset = SimpleNamespace(entries=[{"source": source, "group": "task", "episode_index": episode}
                                           for source in ("a", "b") for episode in range(3) for window in range(2)])
        indices, coverage = select_held_episodes_v69(dataset, 2, 17)
        self.assertEqual(len(indices), 4)
        self.assertEqual(len({(dataset.entries[index]["source"], dataset.entries[index]["episode_index"]) for index in indices}), 4)
        self.assertEqual([row["selected_episodes"] for row in coverage], [2, 2])


if __name__ == "__main__":
    unittest.main()
