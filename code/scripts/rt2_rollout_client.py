"""RoboTwin2 rollout client. Runs in the simulator environment and talks to the separately hosted policy
server over HTTP. Matches the OFFICIAL RoboTwin pi0 eval (policy/pi0/deploy_policy.py): predict a
50-step chunk (= our WIN=50 = pi0_step=50) from one frame, execute the WHOLE chunk (exec_horizon=50, the
open-loop horizon the model was trained on; smaller horizons mis-fire because our action is a per-step
Δqpos trajectory reconstructed by cumsum), then re-observe + re-predict.
30 FPS video: the env's own eval-video writes only ONE frame per take_action (= 16.67 Hz control rate) and
only of the cached self.now_obs. Instead we hook ENV._update_render (called once per 250 Hz sim substep
inside take_action's TOPP loop) and grab the head camera every CAP_EVERY substeps, so the mp4 shows the
REAL inter-step motion at ~30 fps. Inference is untouched (we never change the chunk or the control rate).

  cd /mnt/pfs/xuhaoming/xr-2/RoboTwin && CUDA_VISIBLE_DEVICES=0 \
    ROBOTWIN_PLANNER_BACKEND=curobo /mnt/pfs/xuhaoming/xr-2/.venv/bin/python \
    /mnt/pfs/public/xuhaoming/instruct_gs_world/code/scripts/rt2_rollout_client.py \
    --task grab_roller --seed 100000 --server http://127.0.0.1:9010 \
    --out /mnt/pfs/public/xuhaoming/instruct_gs_world/eval/robotwin2/example --wrist 1
"""
import os, sys, io, json, base64, argparse, hashlib, random, subprocess, time, urllib.request
os.environ.pop("VK_ICD_FILENAMES", None); os.environ.pop("DISPLAY", None)   # SAPIEN headless render fix
import numpy as np, yaml, importlib

RT = os.path.abspath(os.environ.get("ROBOTWIN_ROOT", "/mnt/pfs/xuhaoming/xr-2/RoboTwin"))
PLANNER_BACKEND = os.environ.get("ROBOTWIN_PLANNER_BACKEND", "curobo").strip().lower()
POLICY_SEED_MODE = "environment_seed_replan_v1"
INSTRUCTION_SEED_MODE = "environment_seed_instruction_v1"
if PLANNER_BACKEND != "curobo":
    raise ValueError(f"formal evaluation requires ROBOTWIN_PLANNER_BACKEND=curobo, got {PLANNER_BACKEND!r}")
os.environ["ROBOTWIN_PLANNER_BACKEND"] = PLANNER_BACKEND
os.chdir(RT); sys.path.insert(0, RT); sys.path.insert(0, os.path.join(RT, "description/utils"))
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError
from generate_episode_instructions import generate_episode_descriptions

def _np_b64(a):
    buf = io.BytesIO(); np.save(buf, a); return base64.b64encode(buf.getvalue()).decode()


def query(server, rgb, depth, K, qpos, instr, policy_seed, left=None, right=None):
    payload = {"rgb_b64": _np_b64(rgb), "depth_b64": _np_b64(depth), "K": K.tolist(),
               "qpos": list(map(float, qpos)), "instruction": instr,
               "policy_seed": int(policy_seed)}
    if left is not None:
        payload["left_b64"] = _np_b64(left); payload["right_b64"] = _np_b64(right)   # wrist views
    req = json.dumps(payload).encode()
    r = urllib.request.urlopen(server.rstrip("/") + "/act", data=req, timeout=180)
    out = json.loads(r.read())
    if "error" in out:
        raise RuntimeError("server: " + out["error"])
    if int(out.get("policy_seed", -1)) != int(policy_seed):
        raise RuntimeError(f"server policy seed mismatch: {out.get('policy_seed')} != {policy_seed}")
    return np.asarray(out["action"], np.float32), base64.b64decode(out["viz_png_b64"])


def policy_seed(task, cfg, environment_seed, episode_index, replan_index):
    identity = (f"{POLICY_SEED_MODE}|{task}|{cfg}|{int(environment_seed)}|"
                f"{int(episode_index)}|{int(replan_index)}").encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") & ((1 << 63) - 1)


def instruction_seed(task, cfg, environment_seed, episode_index, instruction_type, description_count):
    identity = (f"{INSTRUCTION_SEED_MODE}|{task}|{cfg}|{int(environment_seed)}|"
                f"{int(episode_index)}|{instruction_type}|{int(description_count)}").encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") & ((1 << 63) - 1)


def planner_runtime(ENV):
    robot = ENV.robot
    communication = bool(getattr(robot, "communication_flag", False))
    if communication:
        left = getattr(robot, "left_proc", None)
        right = getattr(robot, "right_proc", None)
        active = left is not None and right is not None and left.is_alive() and right.is_alive()
        left_name = right_name = "CuroboPlannerProcess"
    else:
        left = getattr(robot, "left_planner", None)
        right = getattr(robot, "right_planner", None)
        active = bool(robot._planner_is_curobo(left) and robot._planner_is_curobo(right))
        left_name, right_name = type(left).__name__, type(right).__name__
    return {
        "backend": PLANNER_BACKEND,
        "curobo_active": active,
        "communication_processes": communication,
        "left_planner": left_name,
        "right_planner": right_name,
        "left_topp_planner": type(getattr(robot, "left_mplib_planner", None)).__name__,
        "right_topp_planner": type(getattr(robot, "right_mplib_planner", None)).__name__,
    }


def _write_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def create_task_runtime(task, cfg):
    with open(f"./task_config/{cfg}.yml") as f:
        args = yaml.safe_load(f)
    args["task_name"] = task; args["task_config"] = cfg; args["ckpt_setting"] = None
    args["data_type"]["depth"] = True; args["data_type"]["pointcloud"] = False
    args["render_freq"] = 0; args["eval_mode"] = True
    with open(CONFIGS_PATH + "_embodiment_config.yml") as f:
        emb = yaml.safe_load(f)
    with open(CONFIGS_PATH + "_camera_config.yml") as f:
        cam = yaml.safe_load(f)
    hct = args["camera"]["head_camera_type"]; args["head_camera_h"] = cam[hct]["h"]; args["head_camera_w"] = cam[hct]["w"]
    rf = emb[args["embodiment"][0]]["file_path"]
    args["left_robot_file"] = rf; args["right_robot_file"] = rf; args["dual_arm_embodied"] = True

    def ecfg(p):
        with open(os.path.join(p, "config.yml")) as f:
            return yaml.safe_load(f)
    args["left_embodiment_config"] = ecfg(rf); args["right_embodiment_config"] = ecfg(rf)
    ENV = getattr(importlib.import_module(f"envs.{task}"), task)()
    return ENV, args


def reset_episode(ENV, args, seed, episode_index, instr=None):
    ENV.setup_demo(now_ep_num=episode_index, seed=seed, is_test=True, **args)
    if instr is not None:
        ENV.set_instruction(instr)


def prepare_episode(task, cfg, start_seed, episode_index, seed_tries, instruction_type,
                    description_count, instr_override):
    """Match RoboTwin's official seed gate: stable scene, expert plan succeeds, then recreate the same seed."""
    env, args = create_task_runtime(task, cfg)
    for seed in range(start_seed, start_seed + seed_tries):
        try:
            reset_episode(env, args, seed, episode_index)
            episode_info = env.play_once()
            solvable = bool(env.plan_success and env.check_success())
            env.close_env()
        except UnStableError as e:
            env.close_env()
            print(f"  seed {seed} unstable: {e}", flush=True)
            continue
        if not solvable:
            print(f"  seed {seed} rejected: expert plan did not solve", flush=True)
            continue

        # Official RoboTwin resets the same task object, preserving task-specific expert metadata.
        reset_episode(env, args, seed, episode_index)
        selected_instruction_seed = instruction_seed(
            task, cfg, seed, episode_index, instruction_type, description_count)
        if instr_override:
            instruction = instr_override
        else:
            random_state = random.getstate()
            random.seed(selected_instruction_seed)
            descriptions = generate_episode_descriptions(task, [episode_info["info"]], description_count)
            choices = descriptions[0][instruction_type]
            instruction = str(random.choice(choices))
            random.setstate(random_state)
        env.set_instruction(instruction=instruction)
        return env, args, seed, instruction, selected_instruction_seed
    raise RuntimeError(f"no expert-solvable stable seed in [{start_seed}, {start_seed + seed_tries})")


def exec_smooth(ENV, action, exec_h):
    """Execute the first exec_h waypoints of the chunk as ONE CONTINUOUS trajectory (vs take_action's
    per-waypoint stop-start TOPP, which jerks + runs ~5x slow). Interpolate [current, wp1..wpK] to the
    250Hz sim rate, finite-diff velocities, and drive the arms+grippers directly with scene.step +
    _update_render per substep (the render hook captures ~30fps). One chunk-step = 60ms = 15 substeps, so
    K waypoints take K/16.67 s (the intended speed)."""
    K = int(min(exec_h, action.shape[0]))
    la = np.asarray(ENV.robot.get_left_arm_jointState()[:-1], np.float64)    # current left arm [6]
    ra = np.asarray(ENV.robot.get_right_arm_jointState()[:-1], np.float64)   # current right arm [6]
    lg = float(ENV.robot.get_left_gripper_val()); rg = float(ENV.robot.get_right_gripper_val())
    cur = np.concatenate([la, [lg], ra, [rg]])                               # [14]
    wp = np.vstack([cur, action[:K].astype(np.float64)])                     # [K+1,14] current + waypoints
    N = max(2, K * 15)                                                       # 250/16.667 = 15 substeps/waypoint
    tw = np.linspace(0.0, 1.0, K + 1); td = np.linspace(0.0, 1.0, N)
    pos = np.stack([np.interp(td, tw, wp[:, d]) for d in range(14)], 1)      # [N,14] dense positions
    vel = np.gradient(pos, axis=0) * 250.0                                   # [N,14] rad/s (dt=1/250)
    for i in range(N):
        if ENV.eval_success or ENV.take_action_cnt >= ENV.step_lim:
            break
        ENV.robot.set_arm_joints(pos[i, :6], vel[i, :6], "left")
        ENV.robot.set_arm_joints(pos[i, 7:13], vel[i, 7:13], "right")
        ENV.robot.set_gripper(float(pos[i, 6]), "left")
        ENV.robot.set_gripper(float(pos[i, 13]), "right")
        ENV.scene.step()
        ENV._update_render()                                                # render hook captures ~30fps here
        if i % 15 == 14:                                                     # one control step elapsed
            ENV.take_action_cnt += 1
            if ENV.check_success():
                ENV.eval_success = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="grab_roller")
    ap.add_argument("--cfg", default="demo_clean")
    ap.add_argument("--seed", type=int, default=100000, help="first RoboTwin environment seed to try")
    ap.add_argument("--seed_tries", type=int, default=100)
    ap.add_argument("--episode_index", type=int, default=0)
    ap.add_argument("--instruction_type", choices=["seen", "unseen"], default="seen")
    ap.add_argument("--description_count", type=int, default=10)
    ap.add_argument("--server", default="http://127.0.0.1:9010")
    ap.add_argument("--out", default="/tmp/rollout")
    ap.add_argument("--instr", default="")
    ap.add_argument("--exec_horizon", type=int, default=50,
                    help="how many steps of the 50-chunk to execute before re-observing+re-predicting. "
                         "50 = full chunk = pi0_step = the open-loop horizon the model was TRAINED on "
                         "(verified to succeed). Smaller values mis-fire: our action is a per-step Δqpos "
                         "trajectory (qpos=anchor+cumsum(Δ)), so executing only the first K then re-anchoring "
                         "discards the trajectory's accumulation (K=1 barely moves the arm).")
    ap.add_argument("--max_queries", type=int, default=600, help="cap on re-plans (safety)")
    ap.add_argument("--viz_every", type=int, default=10, help="save a 3D-pred overlay every N re-plans")
    ap.add_argument("--wrist", type=int, default=0, help="1 = also send left+right wrist rgb (use ONLY with a "
                    "--wrist-trained ckpt + a server started with --wrist 1)")
    ap.add_argument("--smooth", type=int, default=1, help="1 (default) = execute the chunk as ONE continuous "
                    "trajectory (interp to 250Hz + direct drive) -> smooth, runs at intended speed. 0 = old "
                    "per-waypoint take_action (stop-start TOPP, jerky + ~5x slow).")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    started = time.time()
    ENV, args, actual_seed, instr, selected_instruction_seed = prepare_episode(
        a.task, a.cfg, a.seed, a.episode_index, a.seed_tries, a.instruction_type,
        a.description_count, a.instr)
    planner_info = planner_runtime(ENV)
    if not planner_info["curobo_active"]:
        raise RuntimeError(f"formal evaluation did not instantiate CuRobo planners: {planner_info}")
    print(f"[rollout] task={a.task} requested_seed={a.seed} stable_seed={actual_seed} instr={instr!r}", flush=True)

    H, W = args["head_camera_h"], args["head_camera_w"]
    # --- 30 FPS video: hook _update_render (called once per 250 Hz sim substep inside take_action's TOPP
    # loop) and grab the head camera every CAP_EVERY substeps -> the mp4 shows the REAL inter-step motion at
    # ~30 fps. We do NOT use the env's eval-video (it writes 1 cached frame per take_action = 16.67 Hz). ---
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                           "-video_size", f"{W}x{H}", "-framerate", "30", "-i", "-", "-pix_fmt", "yuv420p",
                           "-vcodec", "libx264", "-crf", "23", f"{a.out}/rollout.mp4"], stdin=subprocess.PIPE)
    CAP_EVERY = 8                                                        # ~250 Hz sim / 8 ~= 31 fps capture
    head_cam = ENV.cameras.static_camera_list[ENV.cameras.head_camera_id]  # render ONLY head (3x cheaper than
    rec = {"on": False, "i": 0, "n": 0, "err": None}                      #   update_picture, which does 3 cams)
    orig_ur = ENV._update_render
    def hooked_ur(*aa, **kk):
        out = orig_ur(*aa, **kk)
        if rec["on"]:
            rec["i"] += 1
            if rec["i"] % CAP_EVERY == 0:
                head_cam.take_picture()                                  # video is required; capture errors are fatal
                im = (np.asarray(head_cam.get_picture("Color")) * 255).clip(0, 255).astype(np.uint8)[:, :, :3]
                ff.stdin.write(np.ascontiguousarray(im).tobytes())
                rec["n"] += 1
        return out
    ENV._update_render = hooked_ur
    rec["on"] = True

    obs = ENV.get_obs()                                                  # frame0 for the first 50-step chunk
    ff.stdin.write(np.ascontiguousarray(
        np.asarray(obs["observation"]["head_camera"]["rgb"], np.uint8)).tobytes())
    rec["n"] += 1
    nq = 0
    policy_seeds = []
    while ENV.take_action_cnt < ENV.step_lim and not ENV.eval_success and nq < a.max_queries:
        rgb = np.asarray(obs["observation"]["head_camera"]["rgb"], np.uint8)
        depth = np.asarray(obs["observation"]["head_camera"]["depth"], np.float64)
        K = np.asarray(obs["observation"]["head_camera"]["intrinsic_cv"], np.float32)
        qpos = np.asarray(obs["joint_action"]["vector"], np.float32)
        left = right = None
        if a.wrist:                                                      # send wrist views (needs --wrist server+ckpt)
            left = np.asarray(obs["observation"]["left_camera"]["rgb"], np.uint8)
            right = np.asarray(obs["observation"]["right_camera"]["rgb"], np.uint8)
        sample_seed = policy_seed(a.task, a.cfg, actual_seed, a.episode_index, nq)
        action, viz_png = query(a.server, rgb, depth, K, qpos, ENV.get_instruction(),
                                sample_seed, left, right)                      # 50-step chunk
        policy_seeds.append(sample_seed)
        _arm = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]                    # DEBUG: recovered deploy Δqpos magnitude
        _dq = np.diff(np.vstack([qpos[None].astype(np.float64), action.astype(np.float64)]), axis=0)[:, _arm]
        print(f"  [deploy dq] p95={np.percentile(np.abs(_dq),95):.4f} max={np.abs(_dq).max():.4f} (train/GT p95~0.05)",
              flush=True)
        if nq % a.viz_every == 0:
            open(f"{a.out}/viz_{nq:04d}.png", "wb").write(viz_png)        # 3D-pred overlay
        if a.smooth:
            exec_smooth(ENV, action, a.exec_horizon)                     # ONE continuous trajectory, intended speed
        else:
            for t in range(min(a.exec_horizon, action.shape[0])):        # old per-waypoint stop-start (jerky)
                if ENV.take_action_cnt >= ENV.step_lim or ENV.eval_success:
                    break
                ENV.take_action(action[t], action_type="qpos")          # TOPP substeps -> hook writes ~30 fps
        obs = ENV.get_obs()                                              # frame0 for the next chunk
        nq += 1
        if nq % 2 == 0:
            print(f"  replan {nq}: cnt={ENV.take_action_cnt}/{ENV.step_lim} vid_frames={rec['n']} "
                  f"success={ENV.eval_success}", flush=True)

    rec["on"] = False
    ENV._update_render = orig_ur
    ff.stdin.close(); ff_rc = ff.wait()
    video_path = os.path.join(a.out, "rollout.mp4")
    if ff_rc != 0 or rec["n"] == 0 or not os.path.isfile(video_path) or os.path.getsize(video_path) == 0:
        raise RuntimeError(f"video encoding failed: rc={ff_rc} frames={rec['n']} path={video_path}")
    print(f"[rollout] DONE success={ENV.eval_success} vid_frames={rec['n']} err={rec['err']} "
          f"-> {video_path}", flush=True)
    result = {
        "task": a.task, "cfg": a.cfg, "episode_index": a.episode_index,
        "requested_seed": a.seed, "seed": actual_seed, "instruction_type": a.instruction_type,
        "description_count": a.description_count, "instruction": instr,
        "instruction_seed_mode": INSTRUCTION_SEED_MODE,
        "instruction_seed": selected_instruction_seed,
        "success": bool(ENV.eval_success), "step_count": int(ENV.take_action_cnt),
        "step_limit": int(ENV.step_lim), "replans": nq, "exec_horizon": a.exec_horizon,
        "smooth": bool(a.smooth), "wrist": bool(a.wrist), "video_frames": rec["n"],
        "video_path": os.path.abspath(video_path), "video_bytes": os.path.getsize(video_path),
        "elapsed_sec": round(time.time() - started, 3), "server": a.server,
        "policy_seed_mode": POLICY_SEED_MODE, "policy_seeds": policy_seeds,
        "robotwin_root": RT, "planner_backend": PLANNER_BACKEND,
        "planner_runtime": planner_info,
        "rollout_python": os.path.abspath(sys.executable),
    }
    ENV.close_env()
    _write_json(os.path.join(a.out, "result.json"), result)


if __name__ == "__main__":
    main()
