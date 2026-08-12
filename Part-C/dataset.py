import json
from pathlib import Path
from typing import Optional

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms






def get_train_transform(img_size: int = 128, augment: bool = False) -> transforms.Compose:
    steps = [
        transforms.Resize((img_size, img_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
    ]
    if augment:
        steps.append(transforms.RandomHorizontalFlip())
    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  
    ])
    return transforms.Compose(steps)


def get_val_transform(img_size: int = 128) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def _parse_captions(cap_file: Path, image_paths: list[Path]) -> list[str]:
    
    with open(cap_file, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in ('annotations', 'captions', 'data', 'items', 'records'):
            value = data.get(key)
            if isinstance(value, list):
                data = value
                break

    if isinstance(data, dict):
        captions = []
        for p in image_paths:
            value = data.get(p.name) or data.get(p.stem) or ''
            if isinstance(value, dict):
                cap = (
                    value.get('caption')
                    or value.get('caption_text')
                    or value.get('text')
                    or value.get('sentence')
                    or ''
                )
            else:
                cap = value
            captions.append(cap)
        return captions

    if isinstance(data, list):
        if len(data) == 0:
            return [''] * len(image_paths)

        if isinstance(data[0], str):
            
            assert len(data) == len(image_paths), (
                f"Caption list length {len(data)} != image count {len(image_paths)}"
            )
            return data

        if isinstance(data[0], dict):
            
            lookup: dict[str, str] = {}
            for item in data:
                fname = (
                    item.get('image')
                    or item.get('image_filename')
                    or item.get('file_name')
                    or item.get('filename')
                    or item.get('image_name')
                    or item.get('image_path')
                    or ''
                )
                if not fname and item.get('image_index') is not None:
                    image_index = int(item['image_index'])
                    candidates = [
                        f'{image_index:06d}.png',
                        f'CLEVR_{cap_file.parent.name}_{image_index:06d}.png',
                    ]
                else:
                    candidates = [fname]
                cap = (
                    item.get('caption')
                    or item.get('caption_text')
                    or item.get('text')
                    or item.get('sentence')
                    or ''
                )
                nested = item.get('captions')
                if not cap and isinstance(nested, list) and nested:
                    first = nested[0]
                    if isinstance(first, str):
                        cap = first
                    elif isinstance(first, dict):
                        cap = first.get('caption') or first.get('text') or ''
                for candidate in candidates:
                    if not candidate:
                        continue
                    lookup[Path(candidate).name] = cap
                    lookup[Path(candidate).stem] = cap
            return [lookup.get(p.name) or lookup.get(p.stem) or '' for p in image_paths]

    raise ValueError(f"Unrecognised caption JSON format in {cap_file}")


class CLEVRDataset(Dataset):
    
    _CAPTION_NAMES = {
        'train': [
            'clevr_train_captions.json',
            'clevr_train_caption.json',
            'CLEVR_train_captions.json',
            'CLEVR_train_caption.json',
            'captions_train.json',
            'train_captions.json',
            'captions.json',
        ],
        'val': [
            'clevr_val_captions.json',
            'clevr_val_caption.json',
            'CLEVR_val_captions.json',
            'CLEVR_val_caption.json',
            'captions_val.json',
            'val_captions.json',
            'captions.json',
        ],
    }

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        img_size: int = 128,
        augment: bool = False,
        transform=None,
    ):
        assert split in ('train', 'val'), f"split must be 'train' or 'val', got '{split}'"

        self.split_dir = Path(data_root) / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        self.transform = transform or (
            get_train_transform(img_size, augment=augment) if split == 'train'
            else get_val_transform(img_size)
        )

        
        img_dir = self.split_dir / 'images'
        if not img_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {img_dir}")

        self.image_paths = sorted(
            p for p in img_dir.iterdir()
            if p.suffix.lower() in ('.png', '.jpg', '.jpeg')
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {img_dir}")

        
        cap_file = None
        for name in self._CAPTION_NAMES[split]:
            candidate = self.split_dir / name
            if candidate.exists():
                cap_file = candidate
                break
        if cap_file is None:
            matches = sorted(self.split_dir.glob('*caption*.json'))
            if matches:
                cap_file = matches[0]

        if cap_file is None:
            print(
                f"[Dataset] WARNING: No caption file found in {self.split_dir}. "
                "Using empty strings."
            )
            self.captions = [''] * len(self.image_paths)
        else:
            self.captions = _parse_captions(cap_file, self.image_paths)
            print(f"[Dataset] split={split}  images={len(self.image_paths)}  "
                  f"captions loaded from {cap_file.name}")

        assert len(self.captions) == len(self.image_paths), (
            f"Mismatch: {len(self.captions)} captions vs {len(self.image_paths)} images"
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.transform(img)
        return img, self.captions[idx]






def build_dataloader(
    data_root: str,
    split: str = 'train',
    batch_size: int = 64,
    num_workers: int = 4,
    img_size: int = 128,
    shuffle: Optional[bool] = None,
    drop_last: Optional[bool] = None,
    augment: bool = False,
) -> DataLoader:
    dataset = CLEVRDataset(data_root, split=split, img_size=img_size, augment=augment)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train') if shuffle is None else shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train') if drop_last is None else drop_last,
        persistent_workers=(num_workers > 0),
    )
    print(f"[DataLoader] split={split}  batches={len(loader)}")
    return loader
