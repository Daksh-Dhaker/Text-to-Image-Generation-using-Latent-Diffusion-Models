![](C:%5CUsers%5Csarth%5CAppData%5CRoaming%5Cmarktext%5Cimages%5C2026-04-06-11-42-19-image.png)

## What to read first (before writing any code)

Read these in order: the CLIP paper (Radford et al. 2021), the ViT paper (Dosovitskiy et al. 2021) sections 3.1–3.2, and the DINO paper (Caron et al. 2021) sections 2–3 and Algorithm 1. Also skim "Attention is All You Need" for the text encoder causal masking details. This reading takes 3–4 hours but saves you days of debugging.

---

## File structure to create

```
col775_a2/
├── dataset.py
├── models/
│   ├── __init__.py
│   ├── vit.py          ← shared ViT backbone
│   ├── clip.py
│   └── dino.py
├── train_clip.py
├── train_dino.py
└── probe/
    ├── linear_probe.py
    ├── tsne_viz.py
    └── retrieval.py
```

---

## Step 1 — `dataset.py` → class `CLEVRDataset`

**What it does:** Loads the HuggingFace dataset. For each item it returns the image (as a PIL/tensor) and its caption string. The JSON field `examples` contains image filenames and captions. Since the dataset is on HF, use `datasets` library or download the parquet files and read locally.

**Key design decisions:**

- `__init__(self, split, transform=None)` — load the JSON, store image paths and captions
- `__len__` — return number of examples
- `__getitem__(self, idx)` — return `(image_tensor, caption_string)`
- Keep image loading lazy (load in `__getitem__`, not in `__init__`)

#### Post Implementation Updates: Actually I am updating it to three parts:

```PY
# Part A — CLIP training dataset (image_tensor, caption_str)
class CLEVRCaptionDataset(Dataset):
    #Also one more extra method added
    def get_all_captions(self):        
        return [cap for _,cap in self.samples]
```

```PY
# Part A — DINO training dataset  (image-only, no captions needed)
class CLEVRImageDataset(Dataset):
```

```PY
# Part Aa — Linear-probe dataset
class CLEVRProbeDataset(Dataset):
```

Each class has same 3 member function , you may call them as needed

---

## Step 2 — `dataset.py` → class `SimpleTokenizer`

**What it does:** Builds a vocabulary from all training captions, then encodes/decodes strings to integer token sequences. No pretrained tokenizer — word-level is fine since captions are formulaic.

Functions inside this class:

- `build_vocab(captions)` — iterate all captions, collect unique words, add `[PAD]`, `[BOS]`, `[EOS]` special tokens, assign integer IDs
- `encode(caption, max_len)` → list of ints, padded/truncated to `max_len`, with BOS at start and EOS at end
- `decode(token_ids)` → string, stripping special tokens

Important: fix `max_len` to something like 32–40 (analyze caption lengths in the dataset to choose).

#### Post Implementation Update: Implemented it as describd in plan here.

---

## Step 3 — `dataset.py` → class `DINOAugment`

**What it does:** Produces the multi-crop view for one image — 2 global crops (224×224) and 8 local crops (96×96), each with random resized crop, horizontal flip, color jitter, grayscale, and Gaussian blur (following DINO paper Table 1). Returns a list of 10 tensors.

- `__call__(self, image)` → `List[Tensor]` of length 10, first two are global, rest are local

#### Post Implementation Update: Implemented it as describd in plan here.

---

## Step 4 — `models/vit.py` → class `PatchEmbed`

**What it does:** Splits an image into non-overlapping 16×16 patches and projects each to `hidden_dim=384`. Implemented as a single `nn.Conv2d(3, 384, kernel_size=16, stride=16)`.

- `__init__(self, img_size=224, patch_size=16, embed_dim=384)`
- `forward(self, x)` → shape `(B, N, 384)` where N = (224/16)² = 196

---

## Step 5 — `models/vit.py` → class `TransformerBlock`

**What it does:** One standard ViT block — LayerNorm → MultiHeadAttention → residual → LayerNorm → MLP → residual. This is the shared building block for both CLIP's image encoder and DINO. Use `nn.MultiheadAttention` (allowed) but do NOT use `nn.TransformerEncoderLayer` (not allowed per spec).

- `__init__(self, embed_dim, num_heads, mlp_dim, dropout=0.0)`
- `forward(self, x, mask=None)` — `mask` is needed for the text encoder's causal mask

MLP inside: `Linear(384→1536) → GELU → Linear(1536→384)`.

---

## Step 6 — `models/vit.py` → class `VisionTransformer`

**What it does:** Full ViT = PatchEmbed + learnable `[CLS]` token + learnable positional embeddings + 12 TransformerBlocks + LayerNorm. Returns both the `[CLS]` token and all patch embeddings (you'll need both for the linear probe GAP variant).

- `__init__(self, img_size, patch_size, embed_dim, depth, num_heads, mlp_dim)`
- `forward(self, x)` → `(cls_token, patch_embeddings)` both as tensors

This class is imported and reused by both `clip.py` and `dino.py`.

---

## Step 7 — `models/clip.py` → class `TextTransformer`

**What it does:** 6-layer causal transformer for text. Key difference from ViT: uses a causal attention mask (upper-triangular) so each token only attends to past tokens. Uses the same `TransformerBlock` you already wrote. The representation of the last non-padding token (the EOS position) is taken as the text embedding.

- `__init__(self, vocab_size, max_len, embed_dim, depth, num_heads, mlp_dim)`
- `forward(self, token_ids)` → text embeddings before projection, shape `(B, embed_dim)`

You need token embeddings + positional embeddings + causal mask generation inside here.

---

## Step 8 — `models/clip.py` → class `CLIPModel`

**What it does:** Combines VisionTransformer and TextTransformer. Projects both outputs to 512-d shared space via separate `nn.Linear` layers. Applies L2-norm. Holds the learnable log-temperature scalar.

- `__init__(self, vit, text_encoder, embed_dim=512)`
- `encode_image(self, images)` → L2-normed 512-d image embeddings
- `encode_text(self, token_ids)` → L2-normed 512-d text embeddings
- `forward(self, images, token_ids)` → `(image_embs, text_embs, logit_scale)` — returns all three so the loss function can use them

---

## Step 9 — `models/clip.py` → standalone function `clip_loss`

**What it does:** Implements the symmetric InfoNCE loss from the CLIP paper. Given image embeddings, text embeddings, and the temperature scalar, computes the N×N similarity matrix, then cross-entropy in both directions (image→text and text→image), averaged.

- `clip_loss(image_embs, text_embs, logit_scale)` → scalar loss

This is a standalone helper function, not inside any class.

---

## Step 10 — `train_clip.py` → function `clip_collate_fn`

**What it does:** The spec explicitly says you must handle the case where duplicate captions appear in one batch (since many images share the same caption). This collate function groups the batch and either removes duplicates or marks them so the loss ignores them. One clean approach: for any row where two captions are identical, mask out those pairs from the contrastive loss computation.

- `clip_collate_fn(batch)` → `(image_batch, token_batch, valid_pair_mask)`

---

## Step 11 — `train_clip.py` → function `build_scheduler`

**What it does:** Creates the cosine decay scheduler with linear warmup as specified. Use PyTorch's `LambdaLR` — define a lambda that linearly ramps from 0→1 for the first `warmup_steps`, then follows cosine decay.

- `build_scheduler(optimizer, warmup_steps, total_steps)` → `LambdaLR` scheduler

---

## Step 12 — `train_clip.py` → function `train_clip_one_epoch`

**What it does:** One full pass over the dataloader. For each batch: forward pass through `CLIPModel`, compute `clip_loss`, backprop, clip gradients (important for stability), optimizer step, scheduler step. Returns average loss.

- `train_clip_one_epoch(model, dataloader, optimizer, scheduler, device)` → `float`

---

## Step 13 — `train_clip.py` → function `main`

**What it does:** Entry point. Parses args (epochs, batch size, lr, checkpoint dir), builds dataset/dataloader/model/optimizer/scheduler, runs the training loop, saves checkpoints every N epochs. Also logs loss to a file for plotting.

- `main()` using `argparse`

---

## Step 14 — `models/dino.py` → class `DINOHead`

**What it does:** The projection head on top of the ViT backbone. Takes the 384-d CLS token, passes through an MLP to 4096-d, then applies L2 norm and a weight-normalized final linear layer (this last part is important from the paper for stability).

- `__init__(self, in_dim=384, out_dim=4096, hidden_dim=2048, bottleneck_dim=256)`
- `forward(self, x)` → 4096-d projected embeddings

---

## Step 15 — `models/dino.py` → class `DINOModel`

**What it does:** Wraps a VisionTransformer + DINOHead together. One class is instantiated twice — once as the student, once as the teacher. Teacher is created by deepcopying student and stopping gradient.

- `__init__(self, vit, head)`
- `forward(self, crops)` → projected embeddings for each crop view

---

## Step 16 — `models/dino.py` → function `update_teacher_ema`

**What it does:** Updates teacher parameters as exponential moving average of student. Called after every optimizer step.

- `update_teacher_ema(student, teacher, momentum)` — in-place update, `θ_teacher ← m·θ_teacher + (1−m)·θ_student`

This is a standalone function, not a class method, since it operates on two separate model instances.

---

## Step 17 — `models/dino.py` → class `DINOLoss`

**What it does:** Implements DINO's cross-entropy loss with the centering mechanism. Maintains a running center as an EMA of teacher outputs across batches. Subtracts center from teacher logits before softmax. Computes cross-entropy between teacher (global crop only) and student (all other crops), excluding the matching pair.

- `__init__(self, out_dim, teacher_temp, student_temp, center_momentum)`
- `forward(self, student_outputs, teacher_outputs, n_global_crops, n_local_crops)` → scalar loss
- `update_center(self, teacher_outputs)` — updates the EMA center, called inside `forward`

---

## Step 18 — `train_dino.py` → function `dino_collate_fn`

**What it does:** Each dataset item produces 10 crops (list of tensors). The collate function stacks them into a batch where the first dimension is the crop index, making it easy to slice out global vs local crops during training.

- `dino_collate_fn(batch)` → `List[Tensor]` of length 10, each `(B, 3, H, W)`

---

## Step 19 — `train_dino.py` → function `train_dino_one_epoch`

**What it does:** For each batch, passes all 10 crops through the student, passes only the 2 global crops through the teacher (with `torch.no_grad()`), computes DINOLoss, backprops through student only, clips gradients, steps optimizer, then calls `update_teacher_ema`. Also updates the center.

- `train_dino_one_epoch(student, teacher, loss_fn, dataloader, optimizer, scheduler, device)` → `float`

---

## Step 20 — `train_dino.py` → function `main`

Same pattern as CLIP's main — argparse, build everything, run loop, save checkpoints for both student and teacher.

---

## Step 21 — `probe/linear_probe.py` → function `extract_features`

**What it does:** Loads a frozen model checkpoint, iterates the probe dataset (no gradients), extracts both the CLS embedding and the GAP-over-patches embedding for each image, saves them to disk as numpy arrays for reuse.

- `extract_features(model, dataloader, device)` → `(cls_features, gap_features, labels)` all as numpy arrays

---

## Step 22 — `probe/linear_probe.py` → class `LinearProbe`

**What it does:** A single `nn.Linear(input_dim, num_classes)` with no hidden layers. Trained on frozen features for the counting task (cross-entropy) and color prediction task (binary cross-entropy with sigmoid, since multi-label).

- `__init__(self, input_dim, num_classes)`
- `forward(self, x)` → logits

---

## Step 23 — `probe/linear_probe.py` → function `train_probe` and `eval_probe`

- `train_probe(probe, features, labels, task, epochs, lr)` → trains the linear probe
- `eval_probe(probe, features, labels, task)` → returns accuracy (counting) or F1 (color), filling in Table 1

Run this for all 6 model+embedding combinations (CLIP CLS, CLIP GAP, DINO Student CLS, DINO Student GAP, DINO Teacher CLS, DINO Teacher GAP).

---

## Step 24 — `probe/tsne_viz.py` → function `run_tsne`

**What it does:** Takes the 70K training features, runs sklearn's `TSNE(n_components=2, perplexity=30)` (this is slow — subsample to 10K first for speed), returns 2D coordinates.

- `run_tsne(features, n_subsample=10000)` → `(coords_2d, labels_subset)`

---

## Step 25 — `probe/tsne_viz.py` → function `plot_tsne`

- `plot_tsne(coords, labels, title, save_path)` — matplotlib scatter colored by object count, saved to PDF for report

---

## Step 26 — `probe/retrieval.py` → function `compute_recall_at_k`

**What it does:** Given a query embedding matrix and a gallery embedding matrix, computes cosine similarity, ranks gallery, checks if ground-truth index is in top-K.

- `compute_recall_at_k(query_embs, gallery_embs, ground_truth_indices, k)` → `float`

---

## Step 27 — `probe/retrieval.py` → function `run_retrieval`

**What it does:** Loads frozen CLIP, encodes all validation images and captions, runs image→text and text→image retrieval, reports R@1 and R@3, and prints/saves a few example queries with their top retrieved neighbors.

- `run_retrieval(clip_model, val_dataset, tokenizer, device)` → prints a results table

---

## Build order summary

Start with steps 1–3 (data), then 4–6 (shared ViT backbone — this unlocks both CLIP and DINO), then 7–13 (CLIP end-to-end and training), then 14–20 (DINO), then 21–27 (evaluation). Train CLIP first since it converges more predictably and you can validate your ViT implementation quickly.