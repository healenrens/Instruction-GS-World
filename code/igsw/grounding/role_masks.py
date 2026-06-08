"""Module E (v7-min) — open-vocabulary role-mask grounding teacher.

GroundingDINO (frame0 + role phrases -> boxes) -> SAM2 (boxes -> masks). Produces a
soft per-role 2D mask over frame0; the trainer samples it at each Gaussian's anchor
(u,v) to get per-Gaussian task relevance. Frozen, runs as an async/offline worker
(frame0 only per clip). All via transformers (verified: grounding-dino-tiny +
facebook/sam2.1-hiera-large, both Apache-2.0).
"""

from __future__ import annotations

import numpy as np
import torch

GDINO_REPO = "IDEA-Research/grounding-dino-tiny"
SAM2_REPO = "facebook/sam2.1-hiera-large"


class RoleGrounder:
    def __init__(self, device: str = "cuda", box_thr: float = 0.25, text_thr: float = 0.20):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.device = torch.device(device)
        self.box_thr = box_thr
        self.text_thr = text_thr
        self.gdino_proc = AutoProcessor.from_pretrained(GDINO_REPO)
        self.gdino = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_REPO).to(self.device).eval()
        # SAM2 via transformers
        from transformers import Sam2Processor, Sam2Model
        self.sam_proc = Sam2Processor.from_pretrained(SAM2_REPO)
        self.sam = Sam2Model.from_pretrained(SAM2_REPO).to(self.device).eval()

    @torch.no_grad()
    def detect(self, image: np.ndarray, phrases: list[str]):
        """frame0 [H,W,3] uint8 + phrases -> list of (phrase, box xyxy, score)."""
        from PIL import Image
        pil = Image.fromarray(image).convert("RGB")
        H, W = image.shape[:2]
        text = ". ".join(p.lower().strip() for p in phrases) + "."
        inp = self.gdino_proc(images=pil, text=text, return_tensors="pt").to(self.device)
        out = self.gdino(**inp)
        res = self.gdino_proc.post_process_grounded_object_detection(
            out, inp["input_ids"], threshold=self.box_thr, text_threshold=self.text_thr,
            target_sizes=[(H, W)])[0]
        dets = []
        for box, score, label in zip(res["boxes"], res["scores"], res.get("labels", res.get("text_labels", []))):
            dets.append((str(label), box.detach().cpu().numpy().tolist(), float(score)))
        return dets

    @torch.no_grad()
    def masks_from_boxes(self, image: np.ndarray, boxes: list[list[float]]) -> np.ndarray:
        """boxes [n,4] xyxy -> masks [n,H,W] float in {0,1} via SAM2."""
        from PIL import Image
        if len(boxes) == 0:
            return np.zeros((0, image.shape[0], image.shape[1]), dtype=np.float32)
        pil = Image.fromarray(image).convert("RGB")
        inp = self.sam_proc(images=pil, input_boxes=[[list(map(float, b)) for b in boxes]],
                            return_tensors="pt").to(self.device)
        out = self.sam(**inp)
        masks = self.sam_proc.post_process_masks(
            out.pred_masks.cpu(), inp["original_sizes"].cpu())[0]   # [n, C, H, W] or [n,H,W]
        m = masks.float()
        if m.ndim == 4:                                            # pick best of multimask
            scores = out.iou_scores[0].cpu()                       # [n, C]
            m = m[torch.arange(m.shape[0]), scores.argmax(-1)]
        return (m > 0.5).float().numpy()                           # [n,H,W]

    @torch.no_grad()
    def ground_roles(self, image: np.ndarray, role_phrases: dict[str, str]) -> dict[str, np.ndarray]:
        """role_phrases {role: phrase} -> {role: soft mask [H,W] float}. Background is
        implicit (1 - max(other roles)). Each role mask = union of its detected boxes' SAM2 masks."""
        H, W = image.shape[:2]
        phrases = list(dict.fromkeys(p.lower().strip() for p in role_phrases.values() if p))
        dets = self.detect(image, phrases) if phrases else []
        # map each detection to the role(s) whose phrase contains/matches its label
        role_masks = {r: np.zeros((H, W), np.float32) for r in role_phrases}
        if dets:
            boxes = [d[1] for d in dets]
            sam_masks = self.masks_from_boxes(image, boxes)        # [n,H,W]
            for (label, _, _), msk in zip(dets, sam_masks):
                for role, phr in role_phrases.items():
                    pl = phr.lower().strip()
                    if label and (label in pl or pl in label or any(w in pl for w in label.split())):
                        role_masks[role] = np.maximum(role_masks[role], msk)
        return role_masks
