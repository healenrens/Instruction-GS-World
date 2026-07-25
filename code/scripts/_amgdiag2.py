import sys; sys.path.insert(0, "code"); sys.path.insert(0, "code/scripts")
import numpy as np
from video_gt import load_episode_full
from openvocab_seg import segment_frame_amg


def iou(a, b):
    inter = np.logical_and(a, b).sum(); uni = np.logical_or(a, b).sum()
    return float(inter / uni) if uni else 0.0


for epi in [0, 100, 300]:
    rgb, msk, ooi, instr, n = load_episode_full(epi)
    idm = segment_frame_amg(rgb[0], instr)
    gt = msk[0]
    armIoU = iou(idm == 8, gt == 8)
    bskIoU = iou(idm == 2, gt == 2)
    ovobj = np.isin(idm, [1, 3, 4, 5, 6, 7])
    tcov = float((ovobj & (gt == 1)).sum() / max(1, (gt == 1).sum()))
    comp = {int(i): int((idm == i).sum()) for i in np.unique(idm)}
    print("epi%d: arm=%.2f basket=%.2f target-cov=%.2f | OV id-map: %s" % (epi, armIoU, bskIoU, tcov, comp))
