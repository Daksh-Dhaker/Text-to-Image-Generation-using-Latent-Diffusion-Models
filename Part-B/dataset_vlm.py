import json
import pickle
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
from PIL import Image
from torch.utils.data import Dataset

_STAGE1_SYSTEM = "You are a helpful assistant that describes images."
_STAGE2_SYSTEM = "You are a helpful assistant that answers questions about images."
_IMAGE_PLACEHOLDER = "<image>"
_STAGE2_TARGET_TEMPLATE = "Reasoning: {explanation}\nAnswer: {answer}"

class CaptioningDataset(Dataset):

    _JSON_FILES = {
        "train": "clevr_train_captions.json",
        "val":   "clevr_val_captions.json",
    }

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        assert split in ("train", "val"), \
            f"split must be 'train' or 'val', got '{split}'"

        self.root      = Path(root)
        self.split     = split
        self.transform = transform

        #self.image_dir = self.root / "Clevr_official" / "images" / split
        self.image_dir = self.root / "Clevr_official" / "images" / self._IMAGE_SPLIT[split]
        assert self.image_dir.is_dir(), \
            f"Image dir not found: {self.image_dir}"

        json_path = self.root / "Probe-Datasets" / self._JSON_FILES[split]
        assert json_path.is_file(), \
            f"Caption JSON not found: {json_path}"

        with open(json_path, "r") as f:
            raw: List[dict] = json.load(f)

        self.samples: List[Tuple[str, str]] = [
            (entry["image_filename"], entry["caption"]) for entry in raw
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        filename, caption = self.samples[idx]
        image = Image.open(self.image_dir / filename).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, caption

    def get_all_captions(self) -> List[str]:
        return [cap for _, cap in self.samples]

class QADataset(Dataset):

    _JSON_FILES = {
        "train": "CLEVR_train_explanations_v0.7.10.json",
        # "val":   "CLEVR_val_explanations_v0.7.10.json",
        # dev pkl is carved from train images, so we will use the train JSON for val too
        "val":   "CLEVR_train_explanations_v0.7.10.json",

    }
    _PKL_FILES = {
        "train": "train_images_ids_v0.7.10-recut.pkl",
        "val":   "dev_images_ids_v0.7.10-recut.pkl",
    }
    
    # recut "val" images live in the train image directory
    _IMAGE_SPLIT = {
        "train": "train",
        "val":   "train",
    }

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        training: bool = True,
        use_recut_split: bool = True,
        max_samples: Optional[int] = None,
    ) -> None:
        
        super().__init__()
        assert split in ("train", "val"), \
            f"split must be 'train' or 'val', got '{split}'"

        self.root      = Path(root)
        self.split     = split
        self.transform = transform
        self.training  = training           
        self.image_dir = self.root / "Clevr_official" / "images" / self._IMAGE_SPLIT[split]
        assert self.image_dir.is_dir(), \
            f"Image dir not found: {self.image_dir}"
        allowed_indices: Optional[set] = None
        if use_recut_split:
            pkl_path = self.root / "Probe-Datasets" / self._PKL_FILES[split]
            if pkl_path.is_file():
                with open(pkl_path, "rb") as f:
                    raw_ids = pickle.load(f)
                # pkl is a set of filename strings e.g. 'CLEVR_train_012885.png'
                if isinstance(raw_ids, dict):
                    allowed_indices = set(raw_ids.keys())
                else:
                    allowed_indices = set(raw_ids)
            else:
                import warnings
                warnings.warn(
                    f"CLEVR-X recut pkl not found at {pkl_path}; "
                    "using full JSON without index filtering."
                )

        json_path = self.root / "Probe-Datasets" / self._JSON_FILES[split]
        assert json_path.is_file(), \
            f"Explanations JSON not found: {json_path}"

        with open(json_path, "r") as f:
            data = json.load(f)

        # CLEVR-X JSON is {"info": {...}, "questions": [...]} — unwrap if needed
        if isinstance(data, list):
            raw: List[dict] = data
        else:
            raw: List[dict] = data.get("questions", [])
            if not raw:
                raise RuntimeError(
                    f"Could not find 'questions' list in {json_path}. "
                    f"Top-level keys: {list(data.keys())}"
                )

        self.samples: List[dict] = []
        for entry in raw:
            if not entry.get("factual_explanation"):
                continue
            if allowed_indices is not None:
                if entry["image_filename"] not in allowed_indices:  # pkl contains filenames not int indices
                    continue
            self.samples.append(entry)

        # cap dataset size if requested (useful to limit epoch time)
        if max_samples is not None and max_samples > 0:
            import random as _random
            _random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[object, str, str, str]:
        entry = self.samples[idx]

        image = Image.open(
            self.image_dir / entry["image_filename"]
        ).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        explanations: List[str] = entry["factual_explanation"]
        #explanation = random.choice(explanations) if self.training else explanations[0]
        explanation = explanations[0]

        return (
            image,
            entry["question"],
            explanation,
            str(entry["answer"]),
        )

def format_stage1_prompt() -> str:
    return (
        "<|im_start|>system\n"
        f"{_STAGE1_SYSTEM}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{_IMAGE_PLACEHOLDER}\n"
        "Describe the image."
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def vlm_collate_fn(
    tokenizer,
    max_len: int = 512,
    stage: int = 1,
) -> Callable:

    assert stage in (1, 2), f"stage must be 1 or 2, got {stage}"

    pad_id = tokenizer.pad_token_id
    assert pad_id is not None, \
        "tokenizer.pad_token_id is None — set pad_token before calling vlm_collate_fn"

    IGNORE_IDX = -100

    def _collate_stage1(batch: List[Tuple]) -> Dict[str, torch.Tensor]:
        """batch : List[ (image_tensor, caption_str) ]"""
        images_list, prompt_ids_list, label_ids_list = [], [], []

        prompt_str  = format_stage1_prompt()
        prompt_ids  = tokenizer.encode(prompt_str, add_special_tokens=False)

        for image, caption in batch:
            images_list.append(image)
            cap_ids = (
                tokenizer.encode(caption, add_special_tokens=False)
                + [tokenizer.eos_token_id]
            )
            prompt_ids_list.append(list(prompt_ids))
            label_ids_list.append(cap_ids)

        return _pack_batch(images_list, prompt_ids_list, label_ids_list,
                           pad_id, max_len, IGNORE_IDX)

    def _collate_stage2(batch: List[Tuple]) -> Dict[str, torch.Tensor]:
        images_list, prompt_ids_list, label_ids_list = [], [], []

        for image, question, explanation, answer in batch:
            images_list.append(image)
            prefix_text = (
                f"<|im_start|>system\n{_STAGE2_SYSTEM}<|im_end|>\n"
                f"<|im_start|>user\n{_IMAGE_PLACEHOLDER}\n"
                f"Question: {question}\n<|im_end|>\n"
                f"<|im_start|>assistant\nReasoning: "
            )
            supervised_text = f"{explanation}\nAnswer: {answer}"

            prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
            supervised_ids = (
                tokenizer.encode(supervised_text, add_special_tokens=False)
                + [tokenizer.eos_token_id]
            )
            prompt_ids_list.append(prefix_ids)
            label_ids_list.append(supervised_ids)

        return _pack_batch(images_list, prompt_ids_list, label_ids_list,
                           pad_id, max_len, IGNORE_IDX)

    return _collate_stage1 if stage == 1 else _collate_stage2

def _pack_batch(
    images_list: List,
    prompt_ids_list: List[List[int]],
    label_ids_list: List[List[int]],
    pad_id: int,
    max_len: int,
    ignore_idx: int,
) -> Dict[str, torch.Tensor]:
    input_ids_batch = []
    labels_batch    = []
    attn_mask_batch = []

    for prompt_ids, label_ids in zip(prompt_ids_list, label_ids_list):

        combined_len = len(prompt_ids) + len(label_ids)
        if combined_len > max_len:
            keep_prompt = max_len - len(label_ids)
            if keep_prompt < 0:
                
                label_ids  = label_ids[:max_len]
                prompt_ids = []
            else:
                prompt_ids = prompt_ids[-keep_prompt:]

        seq      = prompt_ids + label_ids
        seq_len  = len(seq)

        label_seq = [ignore_idx] * len(prompt_ids) + list(label_ids)

        pad_len   = max_len - seq_len
        seq       = seq       + [pad_id]     * pad_len
        label_seq = label_seq + [ignore_idx] * pad_len
        attn_mask = [1] * seq_len + [0] * pad_len

        input_ids_batch.append(seq)
        labels_batch.append(label_seq)
        attn_mask_batch.append(attn_mask)

    images = torch.stack(images_list, dim=0)    

    return {
        "images":         images,
        "input_ids":      torch.tensor(input_ids_batch,  dtype=torch.long),
        "attention_mask": torch.tensor(attn_mask_batch,  dtype=torch.long),
        "labels":         torch.tensor(labels_batch,     dtype=torch.long),
    }