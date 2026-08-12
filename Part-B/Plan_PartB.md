![](C:\Users\sarth\AppData\Roaming\marktext\images\2026-04-18-10-51-20-image.png)

## File structure additions for B and C

```
col775_a2/
├── (Part A files unchanged)
├── part_b/
│   ├── dataset_vlm.py
│   ├── models/
│   │   ├── projector.py
│   │   └── vlm.py
│   ├── train_stage1.py
│   ├── train_stage2.py
│   └── eval_vlm.py
```

**Dependency note:** `models/vit.py` and `models/clip.py` from Part A are imported directly. Do not copy them — import them. This means your Part B/C code must sit inside the same repo root or you add the parent path to `sys.path`.

---

## Part B — Vision-Language Modeling

### Step B1 — `part_b/dataset_vlm.py` → class `CaptioningDataset`

**What it does:** Loads `clevr_train_captions.json` (or val variant) for Stage 1. Each item in the JSON maps an image filename to a caption string. Returns `(image_tensor, caption_string)`. Images come from the same `Clevr_official` image directory.

The `__init__` takes a json path, images root, and a transform (same 224×224 normalization as used in Part A). `__getitem__` loads the image and returns the raw caption — tokenization happens in the collate function, not here, so you keep flexibility to swap tokenizers.

---

### Step B2 — `part_b/dataset_vlm.py` → class `QADataset`

**What it does:** Loads `CLEVR_train_explanations_v0.7.10.json` for Stage 2. CLEVR-X format contains fields: `image_filename`, `question`, `answer`, and a list of `factual_explanation` strings. You pick one explanation per sample (randomly during training, deterministically during eval).

Returns `(image_tensor, question_string, explanation_string, answer_string)`. The CoT prompt assembly is handled separately in `format_cot_prompt`.

---

### Step B3 — `part_b/dataset_vlm.py` → function `vlm_collate_fn`

**What it does:** Takes a list of `(image, prompt_tokens, label_tokens)` tuples and produces padded batches. Crucially, the loss must only be computed on label tokens (caption tokens in stage 1, CoT+answer tokens in stage 2), not on the image tokens or system prompt tokens. This function attaches a `label_mask` — a boolean tensor of shape `(B, seq_len)` that is `True` only where the model should be supervised. This is standard LLaVA practice.

- `vlm_collate_fn(batch, pad_token_id)` → `dict` with keys `images`, `input_ids`, `attention_mask`, `labels`

---

### Step B4 — `part_b/models/projector.py` → class `ReverseBNProjector`

**What it does:** A 2-layer MLP that maps each ViT patch token from 384-d to Qwen's hidden dimension (2048-d for Qwen3-4B). "Reverse bottleneck" means it expands first before reaching the target: `Linear(384 → 1536) → GELU → Linear(1536 → 2048)`. The intermediate dimension (1536) is larger than input (384) but smaller than output (2048) — this is the "reverse" of a typical bottleneck.

All 196 patch tokens are projected independently — this is a shared linear layer applied token-wise, not a global pooling.

- `__init__(self, in_dim=384, hidden_dim=1536, out_dim=2048)`
- `forward(self, patch_tokens)` — input shape `(B, 196, 384)`, output `(B, 196, 2048)`

**Part A dependency:** `out_dim` must match Qwen3-4B's `hidden_size`. Verify this by loading Qwen's config: `from transformers import AutoConfig; cfg = AutoConfig.from_pretrained("Qwen/Qwen3-4B-Instruct-2507"); print(cfg.hidden_size)`. Set `out_dim` accordingly (expected: 2048).

---

### Step B5 — `part_b/models/vlm.py` → function `merge_visual_tokens`

**What it does:** Inserts the projected visual tokens into the LLM's input embedding sequence. The standard LLaVA approach replaces a special `<image>` placeholder token in the text prompt with all 196 projected patch embeddings. This function takes the text token embeddings (from Qwen's embedding table) and the projected image tokens, finds the placeholder position, and constructs the merged sequence.

- `merge_visual_tokens(text_embeds, image_tokens, placeholder_idx)` → merged tensor of shape `(B, 196 + text_len - 1, 2048)` and the updated `attention_mask` and `labels`

Keep this as a standalone function — it is called inside `VLMModel.forward` but is easier to test in isolation.

---

### Step B6 — `part_b/models/vlm.py` → class `VLMModel`

**What it does:** The full model. Holds three components: a frozen `VisionTransformer` (loaded from Part A checkpoint), the `ReverseBNProjector`, and a Qwen3 model loaded via HuggingFace `AutoModelForCausalLM`. In Stage 1 Qwen's parameters are also frozen. In Stage 2 LoRA is applied and Qwen's LoRA parameters are unfrozen.

- `__init__(self, vit_checkpoint_path, projector, qwen_model_name="Qwen/Qwen3-4B-Instruct-2507")`
- `forward(self, images, input_ids, attention_mask, labels)` → HuggingFace `CausalLMOutputWithPast` (so loss is computed inside if labels is not None)
- `encode_image(self, images)` → projected patch tokens `(B, 196, 2048)`, useful standalone

**Part A dependency:** Load the ViT from your CLIP or DINO checkpoint. The spec says "best performing vision encoder" — after running Part Aa linear probing, you'll know which one. The `VisionTransformer` class is imported from `models.vit`. Make sure your Part A checkpoint saves the ViT state dict separately (not wrapped inside CLIPModel) so you can load it cleanly here.

---

### Step B7 — `part_b/train_stage1.py` → function `build_stage1_model`

**What it does:** Instantiates `VLMModel`, loads the ViT checkpoint, freezes all ViT parameters with `param.requires_grad = False`, freezes all Qwen parameters the same way, and leaves only the projector trainable. Prints a parameter count summary — the trainable count should be only the projector (~3M params).

- `build_stage1_model(vit_ckpt_path, qwen_name)` → `VLMModel`

---

### Step B8 — `part_b/train_stage1.py` → function `train_stage1_one_epoch`

**What it does:** Standard autoregressive training on the captioning task. For each batch, calls `model.forward`. The `labels` tensor has `-100` everywhere except the caption token positions (HuggingFace's convention for ignoring positions in cross-entropy). Returns average loss.

One subtlety: Qwen uses a specific chat template. For Stage 1, keep the prompt minimal — just a system message like `"Describe the image."` followed by the visual tokens, then the caption as the target.

- `train_stage1_one_epoch(model, dataloader, optimizer, scheduler, tokenizer, device)` → `float`

---

### Step B9 — `part_b/train_stage1.py` → function `main` (Stage 1)

Entry point for Stage 1. Saves checkpoints of the projector weights only (you don't need to re-save the frozen ViT or Qwen). At the end of each epoch, runs a quick validation loop with `torch.no_grad()` to track val loss.

---

### Step B10 — `part_b/train_stage2.py` → function `apply_lora_to_qwen`

**What it does:** Uses `peft` library to wrap Qwen's attention blocks with LoRA adapters. Target modules for Qwen3 attention are typically `q_proj`, `k_proj`, `v_proj`, `o_proj`. Sets rank `r=16`, `lora_alpha=32` (standard 2× scaling), `lora_dropout=0.05`.

- `apply_lora_to_qwen(qwen_model, r=16, target_modules)` → `peft.PeftModel`

After applying LoRA, call `model.print_trainable_parameters()` to confirm only LoRA + projector weights are trainable (should be ~1–2% of total).

---

### Step B11 — `part_b/train_stage2.py` → function `format_cot_prompt`

**What it does:** Given a question, explanation, and answer string, assembles the chain-of-thought training target. The format should be: system prompt + visual tokens placeholder + `"Question: {question}\nReasoning: {explanation}\nAnswer: {answer}"`. The `labels` mask covers only the text after "Question:" — not the system prompt or image placeholder.

- `format_cot_prompt(question, explanation, answer, tokenizer, max_len)` → `(input_ids, labels)`

---

### Step B12 — `part_b/train_stage2.py` → function `train_stage2_one_epoch`

**What it does:** Training loop with three memory-saving techniques required by the spec. Use `torch.cuda.amp.autocast()` and `GradScaler` for mixed-precision. Use gradient accumulation — accumulate over `accum_steps` mini-batches before calling `optimizer.step()`. Enable `model.gradient_checkpointing_enable()` on Qwen to trade compute for memory.

- `train_stage2_one_epoch(model, dataloader, optimizer, scheduler, scaler, tokenizer, device, accum_steps=8)` → `float`

---

### Step B13 — `part_b/train_stage2.py` → function `main` (Stage 2)

Loads the best Stage 1 projector checkpoint, applies LoRA, sets up AdamW with a lower learning rate (try `1e-4` for projector, `2e-5` for LoRA params — use parameter groups). Save LoRA adapter weights with `model.save_pretrained()` and projector separately.

---

### Step B14 — `part_b/eval_vlm.py` → function `generate_response`

**What it does:** Autoregressively generates a response for a given image + prompt using `model.generate()`. Handles the image encoding + token merging before calling generation.

- `generate_response(model, image, prompt, tokenizer, max_new_tokens=200, device)` → `str`

---

### Step B15 — `part_b/eval_vlm.py` → function `compute_exact_match`

Tokenizes predicted and reference strings, strips whitespace/punctuation, returns accuracy over a list of pairs. Used for both Stage 1 (does generated caption exactly match reference?) and Stage 2 (does extracted answer string match gold answer?).

- `compute_exact_match(predictions, references)` → `float`

---

### Step B16 — `part_b/eval_vlm.py` → function `compute_bleu` and `run_eval`

`compute_bleu` wraps `sacrebleu.corpus_bleu` for Stage 1 caption evaluation.

`run_eval` drives the full evaluation: generates outputs for entire train/val split, computes all metrics, prints the table, and saves a PDF of correct and incorrect samples (with the image, prompt, predicted text, and gold text side by side).

---

## Key cross-part compatibility checklist

**B depends on A:**

- `VisionTransformer` from `models/vit.py` must return `(cls_token, patch_tokens)` — your Part A Step 6 already specifies this. Verify the checkpoint saves the ViT state dict with a clean key prefix.
- The image preprocessing (resize to 224×224, normalize with ImageNet stats) must be identical between Part A training and Part B inference — use the same `transform` object.
- After Part Aa, select the best model (CLIP or DINO Teacher) based on Table 1. This choice goes into `build_stage1_model`.

