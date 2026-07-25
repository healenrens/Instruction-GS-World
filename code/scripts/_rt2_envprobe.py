import os, sys, traceback
os.environ.pop("VK_ICD_FILENAMES", None)   # critical: system ICD (api 1.3.242) breaks SAPIEN builtin libvulkan; use bundled
os.environ.pop("DISPLAY", None)

RT  = sys.argv[1] if len(sys.argv) > 1 else "/mnt/pfs/xuhaoming/xr-2/RoboTwin"
TASK= sys.argv[2] if len(sys.argv) > 2 else "grab_roller"
CFG = "demo_clean"
LOGDIR = "/mnt/pfs/public/xuhaoming/instruct_gs_world/logs"

os.chdir(RT)
sys.path.insert(0, RT)
sys.path.insert(0, os.path.join(RT, "description/utils"))
import numpy as np, yaml, importlib, imageio, subprocess
from envs import CONFIGS_PATH

def get_emb_cfg(rf):
    with open(os.path.join(rf, "config.yml")) as f: return yaml.load(f.read(), Loader=yaml.FullLoader)

with open(f"./task_config/{CFG}.yml") as f: args = yaml.load(f.read(), Loader=yaml.FullLoader)
args["task_name"]=TASK; args["task_config"]=CFG; args["ckpt_setting"]=None
args["data_type"]["depth"]=True
args["render_freq"]=0; args["eval_mode"]=True; args["eval_video_log"]=False
with open(os.path.join(CONFIGS_PATH,"_embodiment_config.yml")) as f: emb=yaml.load(f.read(),Loader=yaml.FullLoader)
with open(CONFIGS_PATH+"_camera_config.yml") as f: camcfg=yaml.load(f.read(),Loader=yaml.FullLoader)
et=args["embodiment"]; hct=args["camera"]["head_camera_type"]
args["head_camera_h"]=camcfg[hct]["h"]; args["head_camera_w"]=camcfg[hct]["w"]
rf=emb[et[0]]["file_path"]
args["left_robot_file"]=rf; args["right_robot_file"]=rf; args["dual_arm_embodied"]=True
args["left_embodiment_config"]=get_emb_cfg(rf); args["right_embodiment_config"]=get_emb_cfg(rf)

print("ROOT=",RT,"TASK=",TASK,"emb_file=",rf,flush=True)
print("=== setup_demo(seed=0, is_test=True, depth=True) ===",flush=True)
mod=importlib.import_module(f"envs.{TASK}"); ENV=getattr(mod,TASK)()
ENV.setup_demo(now_ep_num=0, seed=0, is_test=True, **args)
print("SETUP_DEMO_OK step_lim=",ENV.step_lim,flush=True)

obs=ENV.get_obs(); hc=obs["observation"]["head_camera"]
rgb,dep=hc["rgb"],hc["depth"]; K=np.array(hc["intrinsic_cv"]); ext=np.array(hc["extrinsic_cv"]); c2w=np.array(hc["cam2world_gl"])
vec=np.array(obs["joint_action"]["vector"])
valid=dep[dep>0]
print("obs_keys=",list(obs.keys()),flush=True)
print("obs.observation cams=",list(obs["observation"].keys()),flush=True)
print("head_camera keys=",list(hc.keys()),flush=True)
print("joint_action keys=",list(obs["joint_action"].keys()),flush=True)
print("RGB shape/dtype/min/max:",rgb.shape,rgb.dtype,int(rgb.min()),int(rgb.max()),flush=True)
print("DEPTH shape/dtype:",dep.shape,dep.dtype,flush=True)
print("DEPTH valid(mm) min/median/max:",round(float(valid.min()),2),round(float(np.median(valid)),2),round(float(valid.max()),2),flush=True)
print("K=\n",np.array2string(K,precision=3,suppress_small=True),flush=True)
print("extrinsic_cv(3x4 or 4x4)=\n",np.array2string(ext,precision=4,suppress_small=True),flush=True)
print("cam2world_gl=\n",np.array2string(c2w,precision=4,suppress_small=True),flush=True)
print("VECTOR14=",np.array2string(vec,precision=4,suppress_small=True),"shape",vec.shape,flush=True)
print("gripper L(idx6)=",round(float(vec[6]),4)," R(idx13)=",round(float(vec[13]),4),flush=True)
os.makedirs(LOGDIR,exist_ok=True)
imageio.imwrite(f"{LOGDIR}/rt2_probe_rgb.png", rgb)
lo,hi=float(valid.min()),float(np.percentile(valid,99))
dn=np.clip((dep-lo)/max(hi-lo,1e-6)*255,0,255).astype(np.uint8); dn[dep<=0]=0
imageio.imwrite(f"{LOGDIR}/rt2_probe_depth.png", dn)
print("SAVED",f"{LOGDIR}/rt2_probe_rgb.png",f"{LOGDIR}/rt2_probe_depth.png",flush=True)

# ---- video plumbing: open ffmpeg exactly like eval_policy.py ----
vid=f"{LOGDIR}/rt2_probe_video.mp4"
vsize=f"{rgb.shape[1]}x{rgb.shape[0]}"
ENV.eval_video_path=LOGDIR   # take_action writes frames only if this is not None
ff=subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pixel_format","rgb24","-video_size",vsize,
  "-framerate","10","-i","-","-pix_fmt","yuv420p","-vcodec","libx264","-crf","23",vid],stdin=subprocess.PIPE)
ENV._set_eval_video_ffmpeg(ff)

print("=== take_action(current qpos, qpos) x5  + check_success ===",flush=True)
print("before take_action_cnt=",ENV.take_action_cnt,flush=True)
cur=vec.astype(np.float64).copy()
for i in range(5):
    ENV.take_action(cur, action_type="qpos")
    o=ENV.get_obs(); cur=np.array(o["joint_action"]["vector"],dtype=np.float64)
print("after take_action_cnt=",ENV.take_action_cnt," eval_success=",ENV.eval_success," check_success()=",ENV.check_success(),flush=True)
ENV._del_eval_video_ffmpeg()
print("VIDEO_WRITTEN",vid,"size_on_disk=",os.path.getsize(vid),flush=True)
ENV.close_env()
print("=== ALL OK ===",flush=True)
