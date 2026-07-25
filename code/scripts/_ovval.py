import sys; sys.path.insert(0,"code"); sys.path.insert(0,"code/scripts")
import numpy as np
from video_gt import load_episode_full, find_object_id
from openvocab_seg import segment_frame_amg
def iou(a,b):
    u=np.logical_or(a,b).sum(); return float(np.logical_and(a,b).sum()/u) if u else 0.0
for epi in [0,100,200,300,410]:
    rgb,msk,ooi,instr,n=load_episode_full(epi)
    oid,_=find_object_id(msk,n); gt=msk[0]
    ys,xs=np.where(gt==oid); cen=(float(xs.mean()),float(ys.mean())) if xs.size else None
    idm=segment_frame_amg(rgb[0],instr,target_xy=cen)
    tiou=iou(idm==1,gt==oid); aiou=iou(idm==8,gt==8); biou=iou(idm==2,gt==2)
    nobj=len([i for i in np.unique(idm) if 1<=i<=7])
    print("epi%-3d target-IoU=%.2f arm=%.2f basket=%.2f n_obj=%d  obj=%s"%(epi,tiou,aiou,biou,nobj,instr.split(" the ")[-1][:22]))
