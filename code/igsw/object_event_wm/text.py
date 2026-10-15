"""Local frozen instruction features; no video or future enters this encoder."""

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


class FrozenInstructionEncoder(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        from transformers import AutoTokenizer, Qwen3VLModel

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = Qwen3VLModel.from_pretrained(
            model_path, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa", key_mapping={r"^model\.": ""})
        self.width = self.model.config.text_config.hidden_size
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, instructions):
        prompts = [self.tokenizer.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": text}]}],
            tokenize=False, add_generation_prompt=False) for text in instructions]
        tokens = self.tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
        device = self.model.get_input_embeddings().weight.device
        tokens = {name: value.to(device) for name, value in tokens.items()}
        result = self.model(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"],
                            use_cache=False, return_dict=True)
        mask = tokens["attention_mask"].bool()
        return [row[valid].detach().cpu().to(torch.bfloat16)
                for row, valid in zip(result.last_hidden_state, mask)]


def collate_text_features(rows):
    tokens = pad_sequence(rows, batch_first=True)
    lengths = torch.tensor([len(row) for row in rows])
    valid = torch.arange(tokens.shape[1])[None] < lengths[:, None]
    return tokens, valid


class TextFeatureStore:
    def __init__(self, directory):
        self.root = Path(directory)
        self.metadata = json.loads((self.root / "index.json").read_text())
        self.width = self.metadata["width"]
        self.lookup = {text: index for index, text in enumerate(self.metadata["instructions"])}

    def __getitem__(self, instruction):
        return torch.load(self.root / f"{self.lookup[instruction]:07d}.pt", map_location="cpu", weights_only=True)

    def batch(self, instructions, device):
        tokens, valid = collate_text_features([self[text] for text in instructions])
        return tokens.to(device), valid.to(device)
