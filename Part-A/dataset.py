import json
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
import re
import random
import torchvision.transforms as T
import torchvision.transforms.functional as TF

CLEVR_COLORS= ["gray","blue","brown","yellow","red","green","purple","cyan"]
NUM_COLORS = len(CLEVR_COLORS)

# Part A — CLIP training dataset (image_tensor,caption_str)
class CLEVRCaptionDataset(Dataset):
    _CAPTION_FILES = {"train":"clevr_train_captions.json","val":"clevr_val_captions.json"}

    def __init__(self,root,split="train",transform=None):
        super().__init__()
        assert split in ("train","val"),f"split must be 'train' or 'val'"
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.image_dir = self.root/split/"images"
        assert self.image_dir.is_dir(),f"Image directory not found:{self.image_dir}"
        caption_path = self.root/split/self._CAPTION_FILES[split]
        assert caption_path.is_file(),f"Caption JSON not found:{caption_path}"
        with open(caption_path,"r") as f:# Part A caption JSONs are bare lists (no top-level wrapper key)
            raw = json.load(f)
        self.samples = [(entry["image_filename"],entry["caption"]) for entry in raw]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        filename,caption = self.samples[idx]
        image = Image.open(self.image_dir/filename).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image,caption

    def get_all_captions(self):
        return [cap for _,cap in self.samples]

# Part A — DINO training dataset  (image-only,no captions needed)
class CLEVRImageDataset(Dataset):
    _CAPTION_FILES = {"train":"clevr_train_captions.json","val":"clevr_val_captions.json"}
    def __init__(self,root,split= "train",transform= None):
        super().__init__()
        assert split in ("train","val"),f"split must be 'train' or 'val'"
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.image_dir = self.root/split/"images"
        caption_path = self.root/split/self._CAPTION_FILES[split]
        with open(caption_path,"r") as f:
            raw= json.load(f)
        seen = set() # Deduplicate while preserving encounter order
        self.filenames= []
        for entry in raw:
            fn = entry["image_filename"]
            if fn not in seen:
                seen.add(fn)
                self.filenames.append(fn)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self,idx):
        image = Image.open(self.image_dir/self.filenames[idx]).convert("RGB")
        if self.transform is not None:
            return self.transform(image)
        return image

# Part Aa — Linear-probe dataset
class CLEVRProbeDataset(Dataset):
    _PROBE_FILES = {"count":{"train":"clevr_count_train.json","val":"clevr_count_val.json"},"colors":{"train":"clevr_colors_train.json","val":"clevr_colors_val.json"}}

    def __init__(self,root,task= "count",split = "train",transform= None):
        super().__init__()
        assert task in ("count","colors"),f"task must be 'count' or 'colors'"
        assert split in ("train","val"),f"split must be 'train' or 'val'"
        self.root = Path(root)
        self.task = task
        self.split = split
        self.transform = transform
        
        #Image directory:Part_Aa/Clevr_official/images/{split}/
        self.image_dir = self.root/"Clevr_official"/"images"/split
        assert self.image_dir.is_dir(),f"Image directory not found at {self.image_dir}"

        probe_path = (self.root/"Probe-Datasets"/self._PROBE_FILES[task][split])
        assert probe_path.is_file(),f"Probe JSON not found at {probe_path}"
        with open(probe_path,"r") as f:
            data = json.load(f)

        entries= data["examples"]

        #Parse entries
        self.samples= []
        for entry in entries:
            filename = entry["image_filename"] #"image_path" is deliberately not used since it is hardcoded on prof's disk
            if task == "count":
                label = int(entry["label"])# integer class index
            else: #"colors"
                # multi_hot is already [0/1,...] of length 8
                label = torch.tensor(entry["multi_hot"],dtype=torch.float32)
            self.samples.append((filename,label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        filename,label = self.samples[idx]
        image = Image.open(self.image_dir/filename).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image,label

    @property
    def num_classes(self):
        if self.task == "count":
            return max(lbl for _,lbl in self.samples) + 1
        return NUM_COLORS

class CLEVRCaptionDataset_Aa(Dataset):
    _CAPTION_FILES = {"train":"clevr_train_captions.json","val":"clevr_val_captions.json"}

    def __init__(self,root,split="train",transform=None):
        super().__init__()
        assert split in ("train","val"),f"split must be 'train' or 'val'"
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.image_dir = self.root/"Clevr_official"/"images"/split
        assert self.image_dir.is_dir(),f"Image directory not found:{self.image_dir}"
        caption_path = self.root/"Probe-Datasets"/self._CAPTION_FILES[split]
        assert caption_path.is_file(),f"Caption JSON not found:{caption_path}"
        with open(caption_path,"r") as f:# Part A caption JSONs are bare lists (no top-level wrapper key)
            raw = json.load(f)
        self.samples = [(entry["image_filename"],entry["caption"]) for entry in raw]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        filename,caption = self.samples[idx]
        image = Image.open(self.image_dir/filename).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image,caption

    def get_all_captions(self):
        return [cap for _,cap in self.samples]

# SimpleTokenizer — word-level,built from training captions
class SimpleTokenizer:
    PAD_TOKEN = "[PAD]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    UNK_TOKEN = "[UNK]"
    SPECIAL_TOKENS = [PAD_TOKEN,BOS_TOKEN,EOS_TOKEN,UNK_TOKEN]
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3
    DEFAULT_MAX_LEN = 40 # BOS + up to 38 content tokens + EOS
    
    def __init__(self):
        self.word2id: dict = {}
        self.id2word: dict = {}
        self._built = False

    @staticmethod
    def _tokenize(caption):
        tokens = []
        for word in caption.lower().split():
            word = word.strip(",:.")
            if word:
                tokens.append(word)
        return tokens

    def build_vocab(self,captions): #you may get captions using CLEVRCaptionDataset.get_all_captions()
        unique_words= {}
        for cap in captions:
            for tok in self._tokenize(cap):
                unique_words[tok] = None# value is node using as a set

        self.word2id = {tok: i for i,tok in enumerate(self.SPECIAL_TOKENS)}# Assign IDs: specials first (fixed positions),then content words
        for word in unique_words:
            if word not in self.word2id:
                self.word2id[word] = len(self.word2id)

        self.id2word = {i: w for w,i in self.word2id.items()}
        self._built = True

    @property
    def vocab_size(self):
        assert self._built,"Call build_vocab() first!"
        return len(self.word2id)

    def encode(self,caption,max_len = None):
        assert self._built,"Call build_vocab() first!"
        if max_len is None:
            max_len = self.DEFAULT_MAX_LEN
        tokens = self._tokenize(caption)
        ids = [self.word2id.get(t,self.UNK_ID) for t in tokens]

        # Truncate content to leave room for BOS + EOS
        max_content = max_len - 2
        ids = ids[:max_content]

        # Build final sequence
        sequence = [self.BOS_ID] + ids + [self.EOS_ID]

        # Pad to exactly max_len
        pad_len = max_len - len(sequence)
        sequence = sequence + [self.PAD_ID] * pad_len

        assert len(sequence) == max_len
        return sequence

    def decode(self,token_ids):
        assert self._built,"Call build_vocab() first!"
        special_ids = {self.PAD_ID,self.BOS_ID,self.EOS_ID,self.UNK_ID}
        words = [self.id2word.get(i,self.UNK_TOKEN)
            for i in token_ids
            if i not in special_ids]
        return " ".join(words)

    def save(self,path):
        assert self._built,"Call build_vocab() first!"
        with open(path,"w") as f:
            json.dump(self.word2id,f,indent=2)

    @classmethod
    def load(cls,path):
        tok = cls()
        with open(path,"r") as f:
            tok.word2id = json.load(f)
        tok.id2word = {int(i): w for w,i in tok.word2id.items()}
        tok._built = True
        return tok

#DINOAugment: refered main_dino.py at https://github.dev/facebookresearch/dino/tree/main line number 419
class GaussianBlur:
    def __init__(self,sigma_min: float = 0.1,sigma_max: float = 2.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self,img):
        sigma = random.uniform(self.sigma_min,self.sigma_max)
        return TF.gaussian_blur(img,kernel_size=23,sigma=sigma)

class Solarize:
    def __init__(self,threshold= 128):
        self.threshold = threshold

    def __call__(self,img):
        return TF.solarize(img,threshold=self.threshold)

class DINOAugment:
    MEAN = (0.485,0.456,0.406)
    STD  = (0.229,0.224,0.225)

    def __init__(self,n_local_crops= 8):
        self.n_local_crops = n_local_crops

        color_jitter = T.RandomApply([T.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.2,hue=0.1)],p=0.8)
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=self.MEAN,std=self.STD),
        ])

        #Global crop 1: strong blur (p=1.0),no solarize
        self.global_transform_1 = T.Compose([
            T.RandomResizedCrop(224,scale=(0.4,1.0),interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            color_jitter,
            T.RandomGrayscale(p=0.2),
            GaussianBlur(sigma_min=0.1,sigma_max=2.0),# p=1.0 (always)
            normalize,
        ])

        #Global crop 2: weak blur (p=0.1),solarize (p=0.2)
        self.global_transform_2 = T.Compose([
            T.RandomResizedCrop(224,scale=(0.4,1.0),interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            color_jitter,
            T.RandomGrayscale(p=0.2),
            T.RandomApply([GaussianBlur(sigma_min=0.1,sigma_max=2.0)],p=0.1),
            T.RandomApply([Solarize(threshold=128)],p=0.2),
            normalize,
        ])

        #Local crops: medium blur (p=0.5),no solarize
        self.local_transform = T.Compose([
            T.RandomResizedCrop(96,scale=(0.05,0.4),interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            color_jitter,
            T.RandomGrayscale(p=0.2),
            T.RandomApply([GaussianBlur(sigma_min=0.1,sigma_max=2.0)],p=0.5),
            normalize,
        ])

    def __call__(self,image):
        crops=[self.global_transform_1(image),self.global_transform_2(image)]
        crops+=[self.local_transform(image) for _ in range(self.n_local_crops)]
        return crops

# smoke-test
# if __name__ == "__main__":
#     import sys

#     PART_A_ROOT  = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/bigdisk/Others/775A2/my_local_dataset_folder/A2_dataset/Part_A")
#     PART_AA_ROOT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/mnt/bigdisk/Others/775A2/my_local_dataset_folder/A2_dataset/Part_Aa")

#     print("=== CLEVRCaptionDataset (Part A,train) ===")
#     ds_cap = CLEVRCaptionDataset(PART_A_ROOT,split="train")
#     img,cap = ds_cap[0]
#     print(f"  len={len(ds_cap)}")
#     print(f"  caption='{cap[:70]}...'")
#     print(f"  image type={type(img).__name__},size={img.size}")

#     print("\n=== CLEVRImageDataset (Part A,train,deduplicated) ===")
#     ds_img = CLEVRImageDataset(PART_A_ROOT,split="train")
#     img = ds_img[0]
#     print(f"  len={len(ds_img)},image size={img.size}")

#     print("\n=== CLEVRProbeDataset — count (Part Aa,train) ===")
#     ds_cnt = CLEVRProbeDataset(PART_AA_ROOT,task="count",split="train")
#     img,lbl = ds_cnt[0]
#     print(f"  len={len(ds_cnt)},label={lbl} (type={type(lbl).__name__})")
#     print(f"  num_classes={ds_cnt.num_classes}")

#     print("\n=== CLEVRProbeDataset — colors (Part Aa,val) ===")
#     ds_col = CLEVRProbeDataset(PART_AA_ROOT,task="colors",split="val")
#     img,lbl = ds_col[0]
#     print(f"  len={len(ds_col)},label={lbl},shape={lbl.shape}")
#     print(f"  num_classes={ds_col.num_classes}")

#     # Tokenizer checks
#     print("\n=== SimpleTokenizer ===")
#     all_captions = ds_cap.get_all_captions()

#     # Caption length analysis (helps justify DEFAULT_MAX_LEN)
#     lengths = [len(SimpleTokenizer._tokenize(c)) for c in all_captions]
#     print(f"  Caption token-count stats (content only,no BOS/EOS):")
#     print(f"    min={min(lengths)},max={max(lengths)},"
#           f"mean={sum(lengths)/len(lengths):.1f}")
#     pct95 = sorted(lengths)[int(0.95 * len(lengths))]
#     print(f"    95th-percentile={pct95}  →  recommend max_len>={pct95+2}")

#     tok = SimpleTokenizer()
#     tok.build_vocab(all_captions)
#     print(f"  vocab_size={tok.vocab_size}  (including 4 special tokens)")
#     print(f"  special ids: PAD={tok.PAD_ID},BOS={tok.BOS_ID},"
#           f"EOS={tok.EOS_ID},UNK={tok.UNK_ID}")

#     # Round-trip test
#     sample = all_captions[0]
#     ids = tok.encode(sample,max_len=40)
#     decoded = tok.decode(ids)
#     print(f"\n  Original : '{sample}'")
#     print(f"  Encoded  : {ids}")
#     print(f"  Decoded  : '{decoded}'")
#     assert len(ids) == 40,"encode must return exactly max_len tokens"
#     assert ids[0] == tok.BOS_ID,"first token must be BOS"

#     # Truncation test
#     long_cap = " ".join(["word"] * 50)
#     tok.word2id["word"] = len(tok.word2id)
#     ids_long = tok.encode(long_cap,max_len=40)
#     assert len(ids_long) == 40
#     assert ids_long[-1] == tok.PAD_ID or ids_long[-1] == tok.EOS_ID

#     # Save / load round-trip
#     tok.save("/tmp/clevr_vocab.json")
#     tok2 = SimpleTokenizer.load("/tmp/clevr_vocab.json")
#     assert tok2.encode(sample) == tok.encode(sample),"save/load mismatch"
#     print("\n  save/load round-trip: OK")

#     print("\nAll checks passed.")