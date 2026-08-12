## How to Arrange the Data

![Data Arrangement](./Readme_Images/data.png)

## Stage-1 Train
```SH
cd Part-B
python train_stage1.py \
    --data-root /path/to/data \
    --vit-ckpt /path/to/clip_ckpt.pt \
    --vit-kind clip_vit \          # or dino_student / dino_teacher
    --output-dir ./outputs/vlm_stage1 \
    --epochs 5 --batch-size 16 --lr 1e-3
```

## Stage-2 Train

```SH
python train_stage2.py \
    --data-root /path/to/data \
    --vit-ckpt ./outputs/vlm_stage1/vit_clean.pt --vit-kind raw \
    --stage1-ckpt ./outputs/vlm_stage1/checkpoints/best.pt \
    --output-dir ./outputs/vlm_stage2 \
    --epochs 3 --batch-size 2 --accum-steps 8
```

## Eval (Stage-1: reports exact-match + BLEU on train & val)

```SH
python eval_vlm.py --stage stage1 \
    --data-root /path/to/data \
    --vit-ckpt ./outputs/vlm_stage1/vit_clean.pt --vit-kind raw \
    --stage1-ckpt ./outputs/vlm_stage1/checkpoints/best.pt \
    --output-dir ./outputs/eval_stage1
```

## Eval (Stage-2: reports exact-match on train & val, qualitative PDF)

```SH
python eval_vlm.py --stage stage2 \
    --data-root /path/to/data \
    --vit-ckpt ./outputs/vlm_stage1/vit_clean.pt --vit-kind raw \
    --stage1-ckpt ./outputs/vlm_stage1/checkpoints/best.pt \
    --stage2-ckpt ./outputs/vlm_stage2/checkpoints/best \
    --output-dir ./outputs/eval_stage2
```

## Example Usuage

```
python train_stage1.py \
    --data-root /mnt/bigdisk/Others/775A2/my_local_dataset_folder/A2_dataset/Part_Aa \
    --vit-ckpt /mnt/bigdisk/Others/775A2/my_local_dataset_folder/A2_Outputs/clip/checkpoints/best.pt \
    --vit-kind clip_vit \
    --output-dir /mnt/bigdisk/Others/775A2/my_local_dataset_folder/A2_Outputs/vlm_stage1 \
    --epochs 5 --batch-size 2 --lr 1e-3
```

## Partial Outputs

```SH
$ python train_stage1.py \
    --data-root /mnt/bigdisk/Others/775A2/my_local_dataset_folder/A2_dataset/Part_Aa \
    --vit-ckpt /mnt/bigdisk/Others/775A2/my_local_dataset_folder/A2_Outputs/clip/checkpoints/best.pt \
    --vit-kind clip_vit \
    --output-dir /mnt/bigdisk/Others/775A2/my_local_dataset_folder/A2_Outputs/vlm_stage1 \
    --epochs 5 --batch-size 2 --lr 1e-3
[main] <image> token id = 151669
  ViT weights loaded from '/mnt/bigdisk/Others/775A2/my_local_dataset_folder/A2_Outputs/vlm_stage1/vit_clean.pt' (prefix='')
Loading Qwen model:Qwen/Qwen3-4B-Instruct-2507
`torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|█████████████████████| 398/398 [00:00<00:00, 8904.06it/s]
Stage-1 trainable params:4,526,848 / 4,048,660,608 (0.11%)
[build_stage1_model] trainable=4.53M / total=4048.66M (0.112%)
{"epoch": 0, "train_loss": 0.169991, "val_loss": 0.097572, "lr": 0.0009261084118279846, "best_val_loss": 0.097572}
{"epoch": 1, "train_loss": 0.085494, "val_loss": 0.077039, "lr": 0.0006819524684817438, "best_val_loss": 0.077039}
{"epoch": 2, "train_loss": 0.066342, "val_loss": 0.064879, "lr": 0.00036408496388182855, "best_val_loss": 0.064879}
```

## part b eval stage - 1 output

For detailed o/p look into PartB/Outputs/ folder
### Train
```JSON
{
  "n_examples": 2000,
  "exact_match": 0.4005,
  "bleu": 90.27692004922532
}
```
### Val
```JSON
{
  "n_examples": 15000,
  "exact_match": 0.3518,
  "bleu": 88.81449597715088
}
```

## part b eval stage -2 output

```JSON
{
  "val": {
    "n_examples": 6000,
    "exact_match": 0.7353333333333333
  },
  "train": {
    "n_examples": 2000,
    "exact_match": 0.746
  }
}
```


## Issues to look into

pip install peft was needed
<br/>
currently dataset vlm is ampling so it may change order of validation samples
```PY
if max_samples is not None and max_samples > 0:
    import random as _random
    _random.shuffle(self.samples)
    self.samples = self.samples[:max_samples]
```
