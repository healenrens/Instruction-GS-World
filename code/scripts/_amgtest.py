import sys; sys.path.insert(0, "code"); sys.path.insert(0, "code/scripts")
import numpy as np
from video_gt import load_episode_full
from openvocab_seg import segment_frame_amg, per_entity_iou
for epi in [0, 100, 200, 300, 410]:
    rgb, msk, ooi, instr, n = load_episode_full(epi)
    idm = segment_frame_amg(rgb[0], instr)
    gt = msk[0]
    iou = per_entity_iou(idm, gt)
    arm = iou.get("arm", 0.0); bsk = iou.get("basket", 0.0)
    ov_obj = np.isin(idm, [1, 3, 4, 5, 6, 7]); tgt = gt == 1
    tcov = float((ov_obj & tgt).sum() / max(1, tgt.sum()))
    nobj = len([i for i in np.unique(idm) if 1 <= i <= 7])
    print("epi%d: arm=%.2f basket=%.2f | TARGET-covered=%.2f | n_objects=%d" % (epi, arm, bsk, tcov, nobj))
