import sys; sys.path.insert(0,"code"); sys.path.insert(0,"code/scripts")
import numpy as np
from video_gt import load_episode_full
from openvocab_seg import segment_frame_amg
def iou(a,b):
    i=np.logical_and(a,b).sum(); u=np.logical_or(a,b).sum(); return float(i/u) if u else 0.0
for epi in [0,100,200,300,410]:
    rgb,msk,ooi,instr,n=load_episode_full(epi); idm=segment_frame_amg(rgb[0],instr); gt=msk[0]
    ov=np.isin(idm,[1,3,4,5,6,7]); tcov=float((ov&(gt==1)).sum()/max(1,(gt==1).sum()))
    nobj=len([i for i in np.unique(idm) if 1<=i<=7])
    print("epi%d arm=%.2f basket=%.2f target-cov=%.2f n_obj=%d"%(epi,iou(idm==8,gt==8),iou(idm==2,gt==2),tcov,nobj))
