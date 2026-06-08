# Research E3 — External Open-Vocabulary Grounding/Segmentation Pipelines
## Teacher / Fallback for 3DGS Role Masks

> **Task:** Given (frame-0 image + object/role phrases: "apple", "basket", "plate",
> "robot hand"), produce 2D binary or soft masks. These masks are projected onto 3D
> Gaussians via each Gaussian's frame-0 anchor pixel (u,v). Used as TEACHER labels
> or fallback when our trained grounding head is uncertain.
>
> **Server context:** HTTP proxy at `http://10.66.65.186:18000` required for all
> outbound downloads (HuggingFace, GitHub releases). Set before any pip/hf-cli call:
> ```bash
> export HTTP_PROXY=http://10.66.65.186:18000
> export HTTPS_PROXY=http://10.66.65.186:18000
> export HF_HUB_OFFLINE=0  # make sure hub is not locked offline
> ```

---

## 1. Grounding DINO (open-set detector via text)

### What it does
Text prompt → bounding boxes with scores and matched labels.
No masks. Must be combined with SAM/SAM2 to get pixel masks.

### License
**Apache 2.0** (verified: github.com/IDEA-Research/GroundingDINO, HF model cards).

### HuggingFace repos (official, via `transformers`)
| Variant | HF repo | Params | Checkpoint size |
|---------|---------|--------|----------------|
| Tiny    | `IDEA-Research/grounding-dino-tiny` | ~172 M | ~694 MB (`.pth`) |
| Base    | `IDEA-Research/grounding-dino-base` | ~341 M | ~938 MB (`.pth`) |

Legacy weights (original IDEA-Research format): `ShilongLiu/GroundingDINO`
(files: `groundingdino_swint_ogc.pth` 694 MB, `groundingdino_swinb_cogcoor.pth` 938 MB).

### Install (via HuggingFace `transformers` — RECOMMENDED, no custom ops)
```bash
# Needs proxy for HF download
export HTTP_PROXY=http://10.66.65.186:18000
export HTTPS_PROXY=http://10.66.65.186:18000

pip install transformers torch torchvision pillow
# That is all. No custom CUDA extension needed via this route.
```

Alternative (IDEA-Research source — requires compilation):
```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO && pip install -e .
# Downloads weights separately via wget/huggingface_hub
```

### API (transformers path — verified from HF docs)
```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

model_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.float16
)

image = Image.open("frame0.jpg")
# Phrases separated by ". "; must be lowercase and end with "."
text_labels = [["apple", "basket", "plate", "robot hand"]]
# Format: list[list[str]] — one list per image in batch

inputs = processor(images=image, text=text_labels, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    threshold=0.35,
    text_threshold=0.25,
    target_sizes=[(image.height, image.width)],
)
# results[0]["boxes"]  → FloatTensor [N, 4] in [x0, y0, x1, y1] pixel coords
# results[0]["scores"] → FloatTensor [N]
# results[0]["text_labels"] → list[str]
```

**Note on text format:** phrases must be lowercased; the library tokenises each phrase
separately. Separate multiple categories with ". " (e.g., `"apple. basket. plate."`
when using the legacy API; list-of-strings when using `post_process_grounded_object_detection`).

### Speed / VRAM
- Tiny (Swin-T): peak inference VRAM ~0.5–1 GB (fp16), ~50–100 ms per 640px image on A100.
- HF transformers path reported ~10–20% slower than the original source repo due to
  preprocessing differences (verified GitHub issue #31533 in transformers).
- **Training-time overhead:** very low — run async, inference-only, ~1 GB dedicated VRAM.

---

## 2. Grounded-SAM / Grounded-SAM-2

### What it does
Pipeline: GroundingDINO (text → boxes) → SAM / SAM2 (boxes → binary masks).
Grounded-SAM-2 adds **video propagation**: after generating masks on frame 0, SAM2's
video memory tracks them through the clip.

### License
Code: **Apache 2.0** (multiple files) + BSD-3-Clause for SAM2 deps.
**Note:** Grounding DINO 1.5 / DINO-X (newer, closed API on deepdataspace.com) requires
an API token from `https://deepdataspace.com/request_api`. The open-source baseline
using `IDEA-Research/grounding-dino-tiny` (above) does NOT need an API token.

### Repos
- Grounded-SAM-2 (IDEA-Research): https://github.com/IDEA-Research/Grounded-SAM-2
- Grounded-SAM (v1, SAM1): https://github.com/IDEA-Research/Grounded-Segment-Anything

### Install
```bash
export HTTP_PROXY=http://10.66.65.186:18000
export HTTPS_PROXY=http://10.66.65.186:18000

# Python 3.10, CUDA 12.1 recommended
conda create -n gsam2 python=3.10 -y && conda activate gsam2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
cd Grounded-SAM-2

# Install SAM2 (skip CUDA extension if needed)
SAM2_BUILD_CUDA=0 pip install -e ".[notebooks]"

# Install GroundingDINO (custom op — can fail on some envs)
pip install --no-build-isolation -e grounding_dino

# Download checkpoints
cd checkpoints && bash download_ckpts.sh     # SAM2 weights
cd ../gdino_checkpoints && bash download_ckpts.sh  # GroundingDINO weights
```

**Download sizes:**
- SAM2.1-hiera-large: ~900 MB
- SAM2.1-hiera-tiny: ~156 MB
- GroundingDINO-T (swint_ogc.pth): ~694 MB
- GroundingDINO-B (swinb_cogcoor.pth): ~938 MB

All scripts call GitHub releases and HuggingFace Hub → need proxy set.

### Key pipeline (text → frame-0 masks → video propagation)
```python
# Pseudo-code matching Grounded-SAM-2's grounded_sam2_video_demo.py pattern
from sam2.build_sam import build_sam2_video_predictor
from groundingdino.util.inference import load_model, predict

# 1. Detect objects in frame 0
gdino = load_model(config_path, weight_path)
boxes, scores, phrases = predict(gdino, frame0_tensor,
                                 caption="apple . basket . plate . robot hand .",
                                 box_threshold=0.35, text_threshold=0.25)

# 2. Initialise SAM2 video predictor with frame-0 box prompts
predictor = build_sam2_video_predictor(model_cfg, checkpoint)
inference_state = predictor.init_state(video_path=video_dir)
for i, box in enumerate(boxes_xyxy):
    predictor.add_new_points_or_box(inference_state, frame_idx=0,
                                    obj_id=i, box=box)

# 3. Propagate masks through clip
for frame_idx, obj_ids, masks in predictor.propagate_in_video(inference_state):
    # masks[i]: bool array H×W for object obj_ids[i]
    pass
```

---

## 3. SAM2 (standalone, box/point → mask + video propagation)

### What it does
Promptable segmentation: given a point or bounding box on image/video frame 0,
returns a binary mask; `VideoPredictor` propagates the object mask through subsequent
frames via streaming memory (transformer memory bank).

### License
**Apache 2.0** (code, weights, training code). BSD-3-Clause for optional CUDA
post-processing kernels. Verified: github.com/facebookresearch/sam2 LICENSE file.

### HuggingFace repos
| Model | HF repo | Params | File size (approx.) |
|-------|---------|--------|---------------------|
| SAM2.1-tiny | `facebook/sam2.1-hiera-tiny` | 38.9 M | ~156 MB |
| SAM2.1-small | `facebook/sam2.1-hiera-small` | 46 M | ~185 MB |
| SAM2.1-base+ | `facebook/sam2.1-hiera-base-plus` | 80.8 M | ~323 MB |
| SAM2.1-large | `facebook/sam2.1-hiera-large` | 224.4 M | ~900 MB |

No access gating — public downloads, just need the proxy.

### Install
```bash
export HTTP_PROXY=http://10.66.65.186:18000
export HTTPS_PROXY=http://10.66.65.186:18000

# Option A: pip (simplest — uses HF transformers)
pip install transformers torch torchvision

# Option B: Meta's own repo (gives SAM2ImagePredictor / SAM2VideoPredictor directly)
git clone https://github.com/facebookresearch/sam2.git && cd sam2
SAM2_BUILD_CUDA=0 pip install -e .   # skip CUDA extension for simpler install
```

### API — image (box-prompted)
```python
# Via Meta repo
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch, numpy as np

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(frame0_rgb_hwc)  # np.uint8 H×W×3 or PIL
    # boxes: np.ndarray shape [N,4] in xyxy pixel coords
    masks, scores, logits = predictor.predict(box=boxes, multimask_output=False)
    # masks: bool [N, H, W]
```

```python
# Via HF transformers
from transformers import Sam2Processor, Sam2Model
processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")
model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to("cuda")

inputs = processor(images=image, input_boxes=[[box_xyxy]], return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model(**inputs)
masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
```

### API — video propagation
```python
from sam2.build_sam import build_sam2_video_predictor
predictor = build_sam2_video_predictor("sam2.1_hiera_large.yaml", checkpoint)
state = predictor.init_state(video_path="/path/to/frames/")  # dir of .jpg
predictor.add_new_points_or_box(state, frame_idx=0, obj_id=0,
                                box=np.array([x0,y0,x1,y1]))
for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
    ...  # masks: list of bool H×W
```

### Speed / VRAM
- Large @ bfloat16: ~39 FPS video, ~6–8 GB VRAM.
- Tiny @ bfloat16: ~91 FPS, ~2–3 GB VRAM.
- For our use (single frame-0 + short clip ~20–40 frames): large model ~200–500 ms total.

---

## 4. SAM3 — Verified existence + status

### Does it exist?
**Yes, SAM3 is real.** Meta AI released Segment Anything Model 3 on **November 19, 2025**
(arxiv: 2511.16719, repo: github.com/facebookresearch/sam3). SAM3.1 was released
March 27, 2026 (SAM3.1 Object Multiplex with shared-memory multi-object tracking).

### What it does
SAM3 is a unified model that **detects, segments, and tracks** objects using
**open-vocabulary concept prompts** (short text phrases like "yellow school bus",
or image exemplars). Unlike SAM2 (which needs per-instance visual prompts), SAM3
understands text concepts and segments all matching instances simultaneously.
Architecture: 848 M parameters, shared vision encoder, DETR-based detector,
transformer encoder-decoder tracker.

### License
**SAM License** (Meta custom, NOT Apache 2.0). This is a proprietary license
separate from SAM2's Apache 2.0. Allows local use and modification but has
restrictions on commercial deployment. See: github.com/facebookresearch/sam3/blob/main/LICENSE.

### Access
**Gated.** You must request access at `https://huggingface.co/facebook/sam3` and
authenticate with `hf auth login` before downloading weights. Approval is not
instantaneous. 1038lab/sam3 is a community re-upload (unverified license compliance).

### Install
Requires Python 3.12+, PyTorch 2.7+, CUDA 12.6+. Much higher requirements than SAM2.

```bash
git clone https://github.com/facebookresearch/sam3.git && cd sam3
conda create -n sam3 python=3.12 && conda activate sam3
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e .
hf auth login   # must have approved access to facebook/sam3
```

### Recommendation for our use
**Do NOT use SAM3 as primary teacher yet.**
- Gated access (blocks automated setup)
- Higher requirements (Py 3.12, PyTorch 2.7, CUDA 12.6)
- Custom license (legal caution needed)
- Newer/less-tested infrastructure
- Our scenario (single-object manipulation, clear frame-0) does not require SAM3's
  multi-instance open-vocab advantage over GroundedSAM2.

**Revisit if** our scenes have multiple same-class objects (e.g., 3 apples) that
GroundingDINO cannot disambiguate.

---

## 5. Florence-2 (Microsoft)

### What it does
Unified vision-language model with prompt-based task dispatch. Relevant tasks:
- `<OD>`: open-vocabulary object detection → boxes + labels
- `<GROUNDING_CAPTION>` / `<PHRASE_GROUNDING>`: phrase → boxes
- `<REFERRING_EXPRESSION_SEGMENTATION>`: phrase → polygon mask (NOT binary mask;
  outputs polygon vertex coordinates as special tokens `<X><Y>`)
- `<REGION_TO_SEGMENTATION>`: box → polygon mask

### License
**MIT** (verified HF model card `microsoft/Florence-2-large`).

### HuggingFace repos
- `microsoft/Florence-2-base` (~230 M params, ~900 MB)
- `microsoft/Florence-2-large` (~770 M params, ~3 GB)

### Install
```bash
pip install transformers pillow torch torchvision
```

### Brief API
```python
from transformers import AutoProcessor, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large", trust_remote_code=True,
    torch_dtype=torch.float16
).to("cuda")
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large",
                                          trust_remote_code=True)

task = "<REFERRING_EXPRESSION_SEGMENTATION>"
prompt = task + "the apple"
inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda", torch.float16)
generated_ids = model.generate(**inputs, max_new_tokens=1024)
result = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
parsed = processor.post_process_generation(result, task=task, image_size=(W, H))
# parsed["<REFERRING_EXPRESSION_SEGMENTATION>"] -> list of polygons (xy pairs)
```

**Caveats for our use:**
- Segmentation output is **polygon vertices**, not binary mask → must rasterize with
  PIL/cv2 `ImageDraw.polygon()`. Polygons can be coarse (4–20 vertices).
- `<OD>` gives boxes but labels are COCO-vocabulary biased; phrase grounding is more
  flexible.
- Florence-2 is single-phrase-at-a-time; loop over object roles.
- Generally less accurate than GroundedSAM2 on unseen manipulation objects.

**Overall role:** Good as a lightweight fallback (MIT license, single pip install,
no custom ops); not primary teacher.

---

## 6. OWLv2 (Google, open-vocabulary detection only)

### What it does
Zero-shot text-conditioned **object detection** (bounding boxes + scores).
No mask output — must pair with SAM2 to get masks (similar to GroundingDINO).
Query format: list of text phrases per image.

### License
**Apache 2.0** (verified HF model card `google/owlv2-large-patch14`).

### HuggingFace repos
- `google/owlv2-base-patch16` (small, fast)
- `google/owlv2-large-patch14` (~307 M params)
- `google/owlv2-large-patch14-ensemble` (best quality, ensemble of two checkpoints)

### Install
```bash
pip install transformers torch torchvision
```

### Brief API
```python
from transformers import Owlv2Processor, Owlv2ForObjectDetection
processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-large-patch14-ensemble")

texts = [["apple", "basket", "plate", "robot hand"]]
inputs = processor(text=texts, images=image, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model(**inputs)
results = processor.post_process_grounded_object_detection(
    outputs, threshold=0.1, target_sizes=[(image.height, image.width)]
)
# results[0]: {"boxes": ..., "scores": ..., "labels": ...}
```

**Caveats:** Generally lower zero-shot AP than GroundingDINO on complex scenes.
OWLv2 excels at image-conditioned few-shot detection (pass an exemplar image instead
of text). For pure text→box, GroundingDINO is preferred.

---

## 7. Qwen3-VL Native Grounding

### Does Qwen3-VL support bounding-box output?
**Yes, verified.** Qwen2.5-VL and Qwen3-VL both support structured JSON bounding-box
output natively. Documented on HF (Qwen/Qwen2.5-VL-7B-Instruct) and in the
Qwen3-VL Technical Report (arxiv 2511.21631).

### Exact prompt format
```python
# System prompt (optional but improves reliability):
system = "You are a helpful assistant capable of object detection."

# User prompt:
user = "Locate every instance that belongs to the following categories: apple, basket, plate, robot hand. Report bbox coordinates in JSON format."

# Or referring-style:
user = "Detect the apple in the image and output its bounding box as JSON."
```

### Output schema
The model outputs JSON in this format (one dict per detected instance):
```json
{"bbox_2d": [x1, y1, x2, y2], "label": "apple"}
```
- Coordinates are **normalized to 0–1000** (not pixels). Convert to pixels:
  ```python
  x1_px = bbox["bbox_2d"][0] / 1000 * image_width
  y1_px = bbox["bbox_2d"][1] / 1000 * image_height
  x2_px = bbox["bbox_2d"][2] / 1000 * image_width
  y2_px = bbox["bbox_2d"][3] / 1000 * image_height
  ```
- (x1, y1) = top-left, (x2, y2) = bottom-right.
- The model resizes/pads to multiples of 16; coordinates refer to the **display size**,
  so use the image's display dimensions, not the tensor dimensions.

### Full minimal example (Qwen2.5-VL-7B via transformers)
```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info  # from QwenLM/Qwen2.5-VL repo
import torch, json

model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name)

messages = [{"role": "user", "content": [
    {"type": "image", "image": "file:///path/to/frame0.jpg"},
    {"type": "text",
     "text": "Locate every instance that belongs to the following categories: "
             "apple, basket, plate, robot hand. "
             "Output a JSON list, each entry: {\"bbox_2d\": [x1,y1,x2,y2], \"label\": \"name\"}."}
]}]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=512)
output_text = processor.batch_decode(
    [generated_ids[0][inputs.input_ids.shape[1]:]], skip_special_tokens=True
)[0]

detections = json.loads(output_text)  # list of {"bbox_2d": [...], "label": "..."}
```

Then pass boxes to SAM2 image predictor for masks.

### Reliability caveats (important for robot manipulation)
1. **Single-instance bias (documented bug):** The model often returns only one box
   when multiple instances of the same category exist (e.g., 2 apples). GitHub issue
   #1257 in QwenLM/Qwen3-VL / QwenLM/Qwen2.5-VL. Mitigation: ask for "all instances"
   explicitly, or use GroundingDINO instead when multiple objects of same class exist.
2. **Coordinate errors in quantized/llama.cpp versions:** GGUF/llama.cpp inference
   gives inaccurate coordinates (GitHub issue ggml-org/llama.cpp #13694). Use the
   native transformers path.
3. **Image preprocessing scaling:** Coordinates must be rescaled from the 0–1000 range
   using display image dimensions (the image Qwen resizes to), not the original raw size.
4. **Prompt sensitivity:** Without "Output a JSON list" instruction, the model may
   return prose descriptions rather than structured JSON. Always include the JSON
   schema in the prompt.
5. **Qwen3-VL-Seg extension (arxiv 2605.07141, May 2026):** A research paper extends
   Qwen3-VL with pixel-level referring segmentation by treating Qwen's predicted box
   as a structural prior fed into a segmentation decoder. This is NOT part of the
   base Qwen3-VL model — it requires additional decoder training.

---

## 8. Concrete Recommendation — Ranked Pipeline

### Ranking by robustness × install-simplicity

| Rank | Pipeline | Robustness | Install friction | VRAM | Download |
|------|----------|-----------|-----------------|------|----------|
| **1** | **GroundingDINO-T (via transformers) + SAM2.1-large** | High | Very low (pip only) | ~7–9 GB combined | ~1.6 GB |
| **2** | Florence-2-large + SAM2.1-large | Medium-High | Very low (pip only) | ~10 GB combined | ~3.9 GB |
| **3** | Grounded-SAM-2 (full repo) | High + video propagation | Medium (custom ops) | ~7–9 GB | ~1.6 GB |
| **4** | Qwen2.5-VL-7B + SAM2.1-large | Medium | Low (pip, but 15 GB model) | ~20+ GB | ~15 GB |
| **5** | OWLv2 + SAM2.1 | Medium | Very low | ~6 GB | ~1.2 GB |
| **6** | SAM3 | High (concept-native) | High (gated + Py3.12) | >16 GB | Unknown |

---

### Recommended primary pipeline: GroundingDINO-T + SAM2.1-large (via HF transformers)

**Why:**
- Both Apache 2.0 — no license concerns.
- Pure `pip install transformers` — zero custom CUDA ops, no compilation.
- Robust for unambiguous manipulation objects (apple, basket, plate, hand) — these
  are visually distinct; GroundingDINO ECCV 2024 handles them reliably at box_threshold=0.35.
- SAM2.1-large gives accurate boundary masks on frame 0.
- SAM2.1 VideoPredictor propagates masks across the training clip without re-running
  GroundingDINO — amortizes the detection cost.
- Total VRAM: ~8 GB with both models at fp16/bfloat16 on one GPU.
- Total download: ~1.6 GB (need proxy).

**Install (complete, proxy-aware):**
```bash
export HTTP_PROXY=http://10.66.65.186:18000
export HTTPS_PROXY=http://10.66.65.186:18000

pip install transformers torch torchvision pillow accelerate

# Pre-download weights (run once, with proxy):
python - <<'EOF'
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import Sam2Processor, Sam2Model

# GroundingDINO-Tiny (~694 MB)
AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")

# SAM2.1-Large (~900 MB)
Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")
Sam2Model.from_pretrained("facebook/sam2.1-hiera-large")
print("All weights cached.")
EOF
```

**Complete async worker example (image + phrases → role masks dict):**
```python
"""
teacher_grounding_worker.py
Async worker: given frame-0 PIL image + list of role phrases,
returns dict {phrase: binary_mask_HxW (np.bool_)}.

Usage:
  worker = GroundedMaskWorker(device="cuda:1")
  masks = worker.get_masks(frame0_img, ["apple","basket","plate","robot hand"])
"""

import torch
import numpy as np
from PIL import Image
from transformers import (
    AutoProcessor, AutoModelForZeroShotObjectDetection,
    Sam2Processor, Sam2Model
)

class GroundedMaskWorker:
    def __init__(self, device="cuda:1",
                 gdino_id="IDEA-Research/grounding-dino-tiny",
                 sam2_id="facebook/sam2.1-hiera-large"):
        self.device = device
        # Load GroundingDINO
        self.gdino_proc = AutoProcessor.from_pretrained(gdino_id)
        self.gdino = AutoModelForZeroShotObjectDetection.from_pretrained(
            gdino_id, torch_dtype=torch.float16
        ).to(device)
        # Load SAM2
        self.sam2_proc = Sam2Processor.from_pretrained(sam2_id)
        self.sam2 = Sam2Model.from_pretrained(
            sam2_id, torch_dtype=torch.bfloat16
        ).to(device)
        self.gdino.eval(); self.sam2.eval()

    @torch.no_grad()
    def get_masks(self, image: Image.Image, phrases: list[str],
                  box_thresh=0.35, text_thresh=0.25) -> dict:
        """
        Returns dict[phrase -> np.bool_ HxW mask].
        If a phrase is not detected, its mask is all-False.
        """
        H, W = image.height, image.width

        # --- Step 1: GroundingDINO detection ---
        text_labels = [phrases]
        inputs = self.gdino_proc(images=image, text=text_labels,
                                 return_tensors="pt").to(self.device)
        outputs = self.gdino(**inputs)
        results = self.gdino_proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=box_thresh, text_threshold=text_thresh,
            target_sizes=[(H, W)]
        )[0]
        boxes = results["boxes"]         # [N, 4] float tensor xyxy
        text_labs = results["text_labels"]  # list[str]

        # Group boxes by phrase (take highest-score box per phrase)
        phrase_boxes: dict[str, torch.Tensor] = {}
        for box, lab, score in zip(boxes, text_labs, results["scores"]):
            lab_lower = lab.lower().strip()
            if lab_lower not in phrase_boxes or score > phrase_boxes[lab_lower][1]:
                phrase_boxes[lab_lower] = (box, score)

        # --- Step 2: SAM2 mask prediction ---
        role_masks = {}
        for phrase in phrases:
            ph_key = phrase.lower().strip()
            if ph_key not in phrase_boxes:
                role_masks[phrase] = np.zeros((H, W), dtype=bool)
                continue
            box = phrase_boxes[ph_key][0].cpu().numpy()  # [x0,y0,x1,y1]
            input_boxes = [[[box.tolist()]]]  # [batch, n_obj, 4]

            sam_in = self.sam2_proc(images=image, input_boxes=input_boxes,
                                    return_tensors="pt").to(self.device)
            sam_out = self.sam2(**sam_in)
            masks = self.sam2_proc.post_process_masks(
                sam_out.pred_masks.cpu(), sam_in["original_sizes"]
            )[0]  # [1, 1, H, W] bool
            role_masks[phrase] = masks[0, 0].numpy()

        return role_masks
```

**Video propagation (for training clips with multiple frames):**
```python
# Use Meta's native SAM2 API for video propagation
# (HF transformers Sam2VideoModel also works but Meta API is more tested)
from sam2.build_sam import build_sam2_video_predictor

predictor = build_sam2_video_predictor(
    "sam2.1_hiera_large.yaml",
    "/path/to/sam2.1_hiera_large.pt"
)
state = predictor.init_state(video_path="/path/to/clip_frames/")

# Add frame-0 box prompts from GroundingDINO
for obj_id, (phrase, box) in enumerate(phrase_boxes.items()):
    predictor.add_new_points_or_box(
        state, frame_idx=0, obj_id=obj_id,
        box=box.cpu().numpy()
    )

frame_masks = {}  # frame_idx -> dict[phrase -> mask]
for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
    frame_masks[frame_idx] = {
        list(phrase_boxes.keys())[oid]: masks[i][0].cpu().numpy()
        for i, oid in enumerate(obj_ids)
    }
```

---

### Secondary recommendation: Qwen2.5-VL-7B + SAM2.1 (richer semantics, more VRAM)

Use when you need richer semantic understanding (affordances, spatial relations like
"the apple inside the basket") that GroundingDINO cannot express. Requires 15 GB
download and ~20 GB VRAM (use a dedicated GPU).

**Caveat:** Single-instance bias — if a scene has 2 apples, may return only one box.
Workaround: post-process with NMS and explicit "detect ALL instances" in prompt.

---

### Fallback chain for our teacher system

```
Attempt 1: GroundingDINO-T (box_thresh=0.35)
  → if box found: SAM2.1-large (single image) → binary mask

Attempt 2: GroundingDINO-T (lower threshold=0.20)
  → if box found: SAM2.1-large

Attempt 3: Qwen2.5-VL-7B grounding prompt
  → parse JSON bbox → SAM2.1-large

Attempt 4: Manual annotation / skip sample
```

For video clips in the training set, run Attempt 1 once on frame 0 then propagate
with SAM2 VideoPredictor. Avoids re-running detection on every frame.

---

## 9. Proxy setup summary

All downloads need `HTTP_PROXY=http://10.66.65.186:18000`.

| Tool | Download mechanism | Proxy works? |
|------|--------------------|-------------|
| `transformers.from_pretrained()` | HuggingFace Hub (httpx) | Yes — env var `HTTP_PROXY`/`HTTPS_PROXY` |
| `pip install` | pip + PyPI | Yes — env var |
| `bash download_ckpts.sh` (Grounded-SAM-2) | wget/curl | Yes — env var `http_proxy` (lowercase) |
| `hf auth login` + gated models (SAM3) | HF Hub | Yes — same env var |
| GitHub clone | git | `git config --global http.proxy http://10.66.65.186:18000` |

**Important:** `hf_transfer` (Rust-based fast downloader) does NOT support proxies.
Keep it disabled (default): `export HF_HUB_ENABLE_HF_TRANSFER=0`.

---

## 10. Verified sources

- [GroundingDINO GitHub (IDEA-Research/GroundingDINO)](https://github.com/IDEA-Research/GroundingDINO) — Apache 2.0, weights, API
- [GroundingDINO HF docs (transformers)](https://huggingface.co/docs/transformers/model_doc/grounding-dino) — verified Python API
- [IDEA-Research/grounding-dino-tiny (HF)](https://huggingface.co/IDEA-Research/grounding-dino-tiny) — 0.2B, Apache 2.0
- [Grounded-SAM-2 GitHub (IDEA-Research/Grounded-SAM-2)](https://github.com/IDEA-Research/Grounded-SAM-2) — install, pipeline
- [facebookresearch/sam2 GitHub](https://github.com/facebookresearch/sam2) — Apache 2.0, SAM2.1 checkpoints
- [facebook/sam2.1-hiera-large (HF)](https://huggingface.co/facebook/sam2.1-hiera-large) — 0.2B, Apache 2.0
- [SAM3 arxiv 2511.16719](https://arxiv.org/abs/2511.16719) — existence confirmed Nov 2025
- [facebookresearch/sam3 GitHub](https://github.com/facebookresearch/sam3) — SAM License (custom), gated
- [facebook/sam3 HF discussions](https://huggingface.co/facebook/sam3/discussions/68) — access gating confirmed
- [Florence-2-large HF](https://huggingface.co/microsoft/Florence-2-large) — MIT, 770 M params
- [Florence-2 transformers docs](https://huggingface.co/docs/transformers/model_doc/florence2) — `<REFERRING_EXPRESSION_SEGMENTATION>` API
- [OWLv2 HF docs](https://huggingface.co/docs/transformers/model_doc/owlv2) — Apache 2.0
- [google/owlv2-large-patch14-ensemble (HF)](https://huggingface.co/google/owlv2-large-patch14-ensemble) — Apache 2.0
- [Qwen2.5-VL-7B bbox_2d format (HF discussion)](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/discussions/13) — verified coordinate schema
- [QwenLM/Qwen3-VL issue #1257](https://github.com/QwenLM/Qwen3-VL/issues/1257) — single-instance bias documented
- [Grounding Qwen3-VL with SAM2 (debuggercafe.com)](https://debuggercafe.com/grounding-qwen3-vl-detection-with-sam2/) — end-to-end pipeline
- [SAM3.1 Meta blog](https://ai.meta.com/blog/segment-anything-model-3/) — SAM3.1 Object Multiplex March 2026
- [Qwen3-VL-Seg arxiv 2605.07141](https://arxiv.org/abs/2605.07141) — Qwen3-VL pixel segmentation extension (research only)
