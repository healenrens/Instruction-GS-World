"""Ordered manipulation videos from the official CLIPort environment and oracle."""

from copy import deepcopy
from fractions import Fraction
from html import escape
import json
from pathlib import Path
import random

import av
import numpy as np
from PIL import Image, ImageDraw
import pybullet as p

from cliport.environments.environment import Environment
from cliport.tasks.task import Task
from cliport.utils import utils


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value.item()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, default=json_value, indent=2, allow_nan=False))


class OrderedPlacement(Task):
    def __init__(self, order, colors=None):
        super().__init__()
        self.order = tuple(order)
        self.colors = colors
        self.max_steps = len(order)+2
        self.pos_eps = .05
        self.task_completed_desc = "ordered placements completed"

    def reset(self, env):
        super().reset(env)
        colors = self.colors or random.sample(("red", "green", "blue", "yellow", "cyan", "brown"), 6)
        self.entities, bowls, blocks = [], [], []
        for kind, urdf, size, selected, destination in (
                ("bowl", "bowl/bowl.urdf", (.12, .12, 0), colors[:3], bowls),
                ("block", "stacking/block.urdf", (.04, .04, .04), colors[3:], blocks)):
            for color in selected:
                pose = self.get_random_pose(env, size)
                body = env.add_object(urdf, pose, "fixed" if kind == "bowl" else "rigid")
                p.changeVisualShape(body, -1, rgbaColor=utils.COLORS[color]+[1])
                entity = {"body_id": body, "kind": kind, "color": color,
                          "spawn_pose": pose, "urdf": urdf}
                self.entities.append(entity)
                destination.append(entity)
        self.program = []
        for index in self.order:
            actor, target = blocks[index], bowls[index]
            instruction = f"put the {actor['color']} block in the {target['color']} bowl"
            self.goals.append(([(actor["body_id"], (0, None))], np.ones((1, 1)),
                               [target["spawn_pose"]], False, True, "pose", None, 1/len(self.order)))
            self.lang_goals.append(instruction)
            self.program.append({"actor_body_id": actor["body_id"], "target_body_id": target["body_id"],
                                 "actor_color": actor["color"], "target_color": target["color"],
                                 "instruction": instruction})
        self.instruction = ", then ".join(row["instruction"] for row in self.program)


class RecordedEnvironment(Environment):
    def __init__(self, assets, physics_hz=240, capture_hz=5):
        self.recording = False
        self.physics_tick = 0
        self.physics_hz, self.capture_stride = physics_hz, physics_hz//capture_hz
        super().__init__(assets, disp=False, hz=physics_hz)

    def begin_recording(self):
        self.physics_tick = 0
        self.frames, self.masks, self.poses, self.frame_ticks, self.contacts = [], [], [], [], []
        self.body_ids = sorted(entity["body_id"] for entity in self.task.entities)
        self.robot_body_ids = {self.ur5}
        connections = [p.getConstraintInfo(p.getConstraintUniqueId(index)) for index in range(p.getNumConstraints())]
        for _ in connections:
            for connection in connections:
                parent, child = connection[0], connection[2]
                if parent in self.robot_body_ids or child in self.robot_body_ids:
                    self.robot_body_ids.update(body for body in (parent, child) if body >= 0)
        self.camera = deepcopy(self.agent_cams[0])
        self.camera["noise"] = False
        self.recording = True
        self.capture()

    def step_simulation(self):
        super().step_simulation()
        # movej also increments the upstream step_counter; this counter counts physics calls only.
        self.physics_tick += 1
        if self.recording and self.physics_tick % self.capture_stride == 0:
            self.capture()

    def capture(self):
        rgb, _, mask = self.render_camera(self.camera)
        self.frames.append(rgb.copy())
        self.masks.append(mask.copy())
        self.frame_ticks.append(self.physics_tick)
        self.poses.append([(*p.getBasePositionAndOrientation(body)[0],
                            *p.getBasePositionAndOrientation(body)[1]) for body in self.body_ids])
        self.contacts.append(sorted({tuple(sorted((int(row[1]), int(row[2]))))
                                     for row in p.getContactPoints() if row[1] != row[2]}))

    def idle(self, seconds):
        for _ in range(round(seconds*self.physics_hz)):
            self.step_simulation()

    def finish_recording(self):
        if self.frame_ticks[-1] != self.physics_tick:
            self.capture()
        self.recording = False


def save_recording(env, output, events, seed, history_seconds, source_revision, future_seconds, partition, combination):
    output.mkdir(parents=True, exist_ok=True)
    frames = np.stack(env.frames)
    masks = np.stack(env.masks)
    seconds = np.array(env.frame_ticks, dtype=np.float64)/env.physics_hz-history_seconds
    np.savez_compressed(output/"rgb.npz", rgb=frames, times_seconds=seconds)
    past = seconds <= 0
    np.savez_compressed(output/"history.npz", rgb=frames[past], times_seconds=seconds[past])
    with av.open(str(output/"video.mp4"), "w") as video:
        stream = video.add_stream("libx264", rate=Fraction(env.physics_hz, env.capture_stride))
        stream.height, stream.width = frames.shape[1:3]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}
        stream.time_base = stream.codec_context.time_base = Fraction(1, env.physics_hz)
        for rgb, tick in zip(frames, env.frame_ticks):
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts, frame.time_base = tick, Fraction(1, env.physics_hz)
            for packet in stream.encode(frame):
                video.mux(packet)
        for packet in stream.encode():
            video.mux(packet)
    poses = np.asarray(env.poses, dtype=np.float64)
    visible_pixels = np.stack([(masks == body).sum((1, 2)) for body in env.body_ids], 1)
    np.savez_compressed(output/"truth.npz", segmentation=masks, body_ids=env.body_ids,
                        poses_xyz_xyzw=poses, visible_pixels=visible_pixels,
                        times_seconds=seconds, physics_ticks=env.frame_ticks)
    metadata = {"contract": "cliport_ordered_video_truth_v1", "seed": seed,
                "source_revision": source_revision, "instruction": env.task.instruction,
                "program": env.task.program, "entities": env.task.entities, "events": events,
                "official_task_reward": env.task.get_reward(),
                "status": "completed" if env.task.get_reward() >= .99 else "oracle_incomplete",
                "physics_hz": env.physics_hz, "capture_every_physics_steps": env.capture_stride,
                "history_seconds": history_seconds, "frame_count": len(frames),
                "prediction_horizon_seconds": future_seconds, "partition": partition,
                "combination": combination,
                "frame_times_seconds": seconds, "image_hw": list(frames.shape[1:3]),
                "camera": env.camera, "camera_position_convention": "CLIPort camera in world coordinates",
                "segmentation_encoding": "CLIPort uint8 body IDs; 255 is renderer background",
                "entity_body_ids": env.body_ids, "robot_body_id": env.ur5,
                "robot_and_tool_body_ids": sorted(env.robot_body_ids),
                "body_count": p.getNumBodies(), "contacts_by_frame": env.contacts,
                "event_time_meaning": "oracle primitive start/end, not inferred object-motion onset",
                "rgb": "rgb.npz", "rgb_storage": "lossless_uint8_THWC",
                "history_rgb": "history.npz",
                "preview": "video.mp4", "truth": "truth.npz",
                "model_training_inputs": "RGB history and instruction only; no action/pose/mask input"}
    write_json(output/"episode.json", metadata)
    panels = []
    chosen = np.linspace(0, len(frames)-1, 6).round().astype(int)
    for index in chosen:
        rgb = Image.fromarray(frames[index]).resize((320, 240))
        panel = Image.new("RGB", (320, 264), "white")
        panel.paste(rgb)
        ImageDraw.Draw(panel).text((8, 245), f"t={seconds[index]:.3f}s / tick={env.frame_ticks[index]}", fill="black")
        panels.append(panel)
    timeline = Image.new("RGB", (320*len(panels), 264), "white")
    for index, panel in enumerate(panels):
        timeline.paste(panel, (320*index, 0))
    timeline.save(output/"timeline.jpg")
    return metadata


def generate(assets, output, seeds, history_seconds=3., tail_seconds=1., source_revision="",
             future_seconds=10., plan_path=""):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    env = RecordedEnvironment(str(assets))
    records, pairs, sections = [], [], []
    plan = json.loads(Path(plan_path).read_text()) if plan_path else None
    scenes = plan["scenes"] if plan else [
        {"scene_seed": seed, "partition": "engineering", "programs": [
            {"order": [0, 1], "combination": "engineering"},
            {"order": [1, 0], "combination": "engineering"}]} for seed in seeds]
    for scene in scenes:
        seed = scene["scene_seed"]
        prefixes = []
        for program in scene["programs"]:
            order = program["order"]
            random.seed(seed)
            np.random.seed(seed)
            env.seed(seed)
            task = OrderedPlacement(order, scene.get("colors_bowls_then_blocks"))
            env.set_task(task)
            obs = env.reset()
            oracle = task.oracle(env)
            env.begin_recording()
            env.idle(history_seconds)
            prefixes.append(np.stack(env.frames).copy())
            events = []
            for attempt in range(task.max_steps):
                if task.get_reward() >= .99:
                    break
                goal_index = len(task.program)-len(task.goals)
                action = oracle.act(obs, env.info)
                if action is None:
                    events.append({"attempt": attempt, "goal_index": goal_index, "status": "oracle_no_action"})
                    break
                start, before = env.physics_tick, len(task.goals)
                obs, reward, _, _ = env.step(action)
                events.append({"attempt": attempt, "goal_index": goal_index,
                               "actor_body_id": task.program[goal_index]["actor_body_id"],
                               "control_start_tick": start, "control_end_tick": env.physics_tick,
                               "goal_completed": len(task.goals) < before, "reward": reward})
            env.idle(max(tail_seconds, history_seconds+future_seconds-env.physics_tick/env.physics_hz))
            env.finish_recording()
            name = f"scene_{seed:06d}_order_{''.join(map(str, order))}"
            metadata = save_recording(env, root/name, events, seed, history_seconds, source_revision,
                                      future_seconds, scene["partition"], program["combination"])
            records.append({"case_id": name, "episode": f"{name}/episode.json", "status": metadata["status"],
                            "scene_seed": seed, "partition": scene["partition"], "combination": program["combination"]})
            print(json.dumps({"event": "simulated_episode", "case_id": name,
                              "status": metadata["status"], "reward": metadata["official_task_reward"],
                              "frames": metadata["frame_count"], "seconds": float(metadata["frame_times_seconds"][-1])}), flush=True)
            sections.append(f'<section><h2>{escape(name)}</h2><p>{escape(task.instruction)}</p>'
                            f'<p>Status: {metadata["status"]}; reward: {metadata["official_task_reward"]}</p>'
                            f'<img src="{name}/timeline.jpg" alt="Actual simulator timeline">'
                            f'<video src="{name}/video.mp4" controls preload="none"></video>'
                            f'<p><a href="{name}/episode.json">Entities, events and camera</a></p></section>')
        pairs.append({"scene_seed": seed, "history_frames": len(prefixes[0]), "programs": len(prefixes),
                      "identical_history_rgb": bool(np.array_equal(prefixes[0], prefixes[1])) if len(prefixes) == 2 else None})
    p.disconnect()
    report = {"contract": "cliport_ordered_video_collection_v1", "source_revision": source_revision,
              "episodes": records, "paired_initial_scenes": pairs,
              "completed_episodes": sum(row["status"] == "completed" for row in records),
              "partition": "scene_disjoint_plan" if plan else "engineering_collection_no_train_test_claim",
              "plan": plan, "history_seconds": history_seconds, "future_seconds": future_seconds,
              "physics_engine": "PyBullet through official CLIPort Environment and Task.oracle"}
    write_json(root/"manifest.json", report)
    (root/"index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Ordered CLIPort Episodes</title>'
        '<style>body{font:16px system-ui;margin:24px;color:#202124;background:white;letter-spacing:0}'
        'section{padding:24px 0;border-top:1px solid #ddd}img{width:100%}video{width:640px;max-width:100%}'
        'h2{font-size:20px;overflow-wrap:anywhere}</style><h1>Ordered CLIPort Episodes</h1>'+''.join(sections)+'</html>')
    return report
