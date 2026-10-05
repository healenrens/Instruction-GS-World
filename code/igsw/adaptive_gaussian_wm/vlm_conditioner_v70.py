"""Local official Qwen3-VL base model for history/instruction conditioning.

API references: transformers/models/qwen3_vl/{modeling,processing,
video_processing}_qwen3_vl.py (Transformers 4.57.1). No LM head or LM loss.
"""
from __future__ import annotations

import torch
from torch import nn


def parameter_counts(module: nn.Module) -> dict[str, int]:
    """Count instantiated parameters, not configuration estimates."""
    return {
        "total": sum(p.numel() for p in module.parameters()),
        "trainable": sum(p.numel() for p in module.parameters() if p.requires_grad),
        "frozen": sum(p.numel() for p in module.parameters() if not p.requires_grad),
    }


class VLMConditionerV70(nn.Module):
    """Freeze vision (including all mergers); fully train the text base model.

    ``prepare_inputs`` accepts rgb [B,16,3,H,W] uint8, pixel_valid
    [B,16,H,W], native_hw [B,2], times [B,16] in seconds and instruction
    list[str]. Frames are already sampled by the parent runtime. It crops
    batch padding only; the official video processor resizes the native clip.
    Text budget includes chat delimiters and official video timestamps. It is
    a soft budget: full instructions survive and overflow is reported.
    Move this module to the rank's device before calling prepare_inputs.
    """

    hidden_size = 2560

    def __init__(self, model_path, visual_tokens=4096, text_tokens=512):
        super().__init__()
        from transformers import AutoProcessor, Qwen3VLModel

        self.visual_tokens = visual_tokens
        self.text_tokens = text_tokens
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = Qwen3VLModel.from_pretrained(
            model_path, local_files_only=True, dtype=torch.bfloat16,
            # Official Instruct checkpoints nest base weights under model.*.
            attn_implementation="sdpa", key_mapping={r"^model\.": ""},
        )
        self.model.visual.requires_grad_(False)
        self.model.visual.eval()
        self.model.language_model.requires_grad_(True)
        self.model.config.use_cache = False
        self.model.language_model.config.use_cache = False
        self.model.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def train(self, mode=True):
        super().train(mode)
        self.model.visual.eval()
        return self

    @property
    def text_blocks(self):
        return self.model.language_model.layers

    @property
    def fsdp_wrap_classes(self):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer

        return (Qwen3VLTextDecoderLayer,)

    @property
    def fsdp_ignored_modules(self):
        return (self.model.visual,)

    @property
    def device(self):
        return self.model.get_input_embeddings().weight.device

    def prepare_inputs(self, batch) -> dict[str, torch.Tensor]:
        from transformers.video_utils import VideoMetadata

        videos, metadata, prompts = [], [], []
        video_processor = self.processor.video_processor
        patch = video_processor.patch_size
        merge = video_processor.merge_size
        temporal = video_processor.temporal_patch_size
        # One merged visual token covers temporal * (patch * merge)^2 pixels.
        pixel_budget = self.visual_tokens * temporal * (patch * merge) ** 2
        for item, instruction in enumerate(batch["instruction"]):
            h, w = batch["native_hw"][item].tolist()
            rgb = batch["rgb"][item, :, :, :h, :w]
            valid = batch["pixel_valid"][item, :, :h, :w]
            videos.append(rgb.masked_fill(~valid[:, None].bool(), 0).contiguous())
            times = batch["times"][item].detach().double().cpu()
            relative_times = (times - times[0]).tolist()
            # fps=1 makes frames_indices/fps equal the supplied timestamps,
            # also for non-uniform sampling. The official processor averages
            # each temporal pair and renders its <... seconds> placeholder.
            metadata.append(VideoMetadata(
                total_num_frames=len(relative_times), fps=1.0,
                frames_indices=relative_times, height=h, width=w,
                duration=relative_times[-1],
            ))
            prompts.append(self.processor.apply_chat_template(
                [{"role": "user", "content": [
                    {"type": "video"}, {"type": "text", "text": instruction},
                ]}], tokenize=False, add_generation_prompt=False,
            ))
        inputs = self.processor(
            text=prompts, videos=videos, padding=True, truncation=False,
            return_tensors="pt", return_mm_token_type_ids=True,
            videos_kwargs={
                "video_metadata": metadata, "do_sample_frames": False,
                "size": {"shortest_edge": min(video_processor.size["shortest_edge"], pixel_budget),
                         "longest_edge": pixel_budget},
            },
        )
        ids, mask = inputs["input_ids"], inputs["attention_mask"].bool()
        visual_counts = ((ids == self.processor.video_token_id) & mask).sum(-1)
        text_counts = mask.sum(-1) - visual_counts
        inputs.update({
            "visual_token_counts": visual_counts,
            "text_token_counts": text_counts,
            "visual_budget_overflow": (visual_counts - self.visual_tokens).clamp_min(0),
            "text_budget_overflow": (text_counts - self.text_tokens).clamp_min(0),
        })
        return {name: inputs[name].to(self.device, non_blocking=True) for name in (
            "input_ids", "attention_mask", "pixel_values_videos", "video_grid_thw", "mm_token_type_ids",
            "visual_token_counts", "text_token_counts", "visual_budget_overflow", "text_budget_overflow",
        )}

    def forward(self, vlm_inputs) -> dict[str, torch.Tensor]:
        model_inputs = {name: vlm_inputs[name] for name in (
            "input_ids", "attention_mask", "pixel_values_videos", "video_grid_thw", "mm_token_type_ids",
        )}
        output = self.model(**model_inputs, use_cache=False, return_dict=True)
        return {"last_hidden_state": output.last_hidden_state,
                "attention_mask": vlm_inputs["attention_mask"]}

    def parameter_inventory(self):
        text = self.model.language_model
        return {
            "conditioner": parameter_counts(self),
            "vision": parameter_counts(self.model.visual),
            "vision_merger": parameter_counts(self.model.visual.merger),
            "vision_deepstack_mergers": parameter_counts(self.model.visual.deepstack_merger_list),
            "text": parameter_counts(text),
            "text_embeddings": parameter_counts(text.embed_tokens),
            "text_blocks": parameter_counts(text.layers),
            "text_norm": parameter_counts(text.norm),
        }
