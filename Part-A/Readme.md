# Downloading the Dataset

Info: Dataset is roughly around 34GB

Step-0: Make a Hugging Face Account

Step-1: Login and generate token at https://huggingface.co/settings/tokens

Step-2: 
Run the dataset download code but make sure you change the download location in code 
```PY
local_dir = "./my_local_dataset_folder"
```

```
pip install huggingface_hub
python download_dataset.py
```

Step-3:

```
cd ./my_local_dataset_folder
cat archive.tar.part* | tar -xf -
```

# Info:

Any updates in plan, please write/refer to partA+Aa.md

# Running Main script
## Train
```SH
python main.py train-clip --data-root ./data/partA --output-dir ./outputs/clip
python main.py train-dino --data-root ./data/partA --output-dir ./outputs/dino
```
## Evaluate everything at once
```SH
python main.py run-all \
    --probe-data-root ./data/partAa \
    --data-root       ./data/partA  \
    --clip-ckpt  ./outputs/clip/checkpoints/best.pt \
    --dino-ckpt  ./outputs/dino/checkpoints/last.pt \
    --tokenizer  ./outputs/clip/tokenizer.json \
    --output-dir ./outputs

```


# Partials Outputs/Results

## Train clip outputs

```SH
$ python main.py train-clip --data-root /mnt/bigdisk/Others/775A2/my_local_dataset_folder/A2_dataset/Part_A --output-dir /mnt/bigdisk/Others/775A2/my_local_dataset_folder/A2_Outputs/clip
{"epoch": 0, "train_loss": 1.249606, "val_loss": 0.235733, "lr": 0.00010012787723785167, "best_val_loss": 0.235733}
{"epoch": 1, "train_loss": 0.163929, "val_loss": 0.104386, "lr": 0.00020012787723785167, "best_val_loss": 0.104386}
{"epoch": 2, "train_loss": 0.077562, "val_loss": 0.057475, "lr": 0.00030012787723785166, "best_val_loss": 0.057475}
{"epoch": 3, "train_loss": 0.053243, "val_loss": 0.045225, "lr": 0.00040012787723785165, "best_val_loss": 0.045225}
{"epoch": 4, "train_loss": 0.044483, "val_loss": 0.033344, "lr": 0.0005, "best_val_loss": 0.033344}
{"epoch": 5, "train_loss": 0.038298, "val_loss": 0.057607, "lr": 0.0004998633143352315, "best_val_loss": 0.033344}
{"epoch": 6, "train_loss": 0.030573, "val_loss": 0.029298, "lr": 0.0004994534068046936, "best_val_loss": 0.029298}
{"epoch": 7, "train_loss": 0.027499, "val_loss": 0.022264, "lr": 0.0004987707256362529, "best_val_loss": 0.022264}
{"epoch": 8, "train_loss": 0.023421, "val_loss": 0.015949, "lr": 0.0004978160173317438, "best_val_loss": 0.015949}
{"epoch": 9, "train_loss": 0.020777, "val_loss": 0.028269, "lr": 0.0004965903258506806, "best_val_loss": 0.015949}
{"epoch": 10, "train_loss": 0.017604, "val_loss": 0.013736, "lr": 0.0004950949914687023, "best_val_loss": 0.013736}
{"epoch": 11, "train_loss": 0.016384, "val_loss": 0.025963, "lr": 0.0004933316493120015, "best_val_loss": 0.013736}
{"epoch": 12, "train_loss": 0.015458, "val_loss": 0.016623, "lr": 0.0004913022275693372, "best_val_loss": 0.013736}
{"epoch": 13, "train_loss": 0.014097, "val_loss": 0.009874, "lr": 0.0004890089453835894, "best_val_loss": 0.009874}
{"epoch": 14, "train_loss": 0.013176, "val_loss": 0.010594, "lr": 0.00048645431042515866, "best_val_loss": 0.009874}
{"epoch": 15, "train_loss": 0.011627, "val_loss": 0.007342, "lr": 0.0004836411161498652, "best_val_loss": 0.007342}
{"epoch": 16, "train_loss": 0.011236, "val_loss": 0.011938, "lr": 0.0004805724387443462, "best_val_loss": 0.007342}
{"epoch": 17, "train_loss": 0.011474, "val_loss": 0.008754, "lr": 0.00047725163376229063, "best_val_loss": 0.007342}
{"epoch": 18, "train_loss": 0.010718, "val_loss": 0.011116, "lr": 0.0004736823324551909, "best_val_loss": 0.007342}
{"epoch": 19, "train_loss": 0.010284, "val_loss": 0.010934, "lr": 0.00046986843780162223, "best_val_loss": 0.007342}
{"epoch": 20, "train_loss": 0.010155, "val_loss": 0.010473, "lr": 0.0004658141202393935, "best_val_loss": 0.007342}
{"epoch": 21, "train_loss": 0.010014, "val_loss": 0.010271, "lr": 0.00046152381310523384, "best_val_loss": 0.007342}
{"epoch": 22, "train_loss": 0.010302, "val_loss": 0.009653, "lr": 0.000457002207787005, "best_val_loss": 0.007342}
{"epoch": 23, "train_loss": 0.009418, "val_loss": 0.006615, "lr": 0.0004522542485937369, "best_val_loss": 0.006615}
{"epoch": 24, "train_loss": 0.009347, "val_loss": 0.006193, "lr": 0.00044728512734909845, "best_val_loss": 0.006193}
{"epoch": 25, "train_loss": 0.008914, "val_loss": 0.009009, "lr": 0.0004421002777142148, "best_val_loss": 0.006193}
{"epoch": 26, "train_loss": 0.008747, "val_loss": 0.006054, "lr": 0.0004367053692460385, "best_val_loss": 0.006054}
{"epoch": 27, "train_loss": 0.008046, "val_loss": 0.006592, "lr": 0.0004311063011977723, "best_val_loss": 0.006054}
{"epoch": 28, "train_loss": 0.008325, "val_loss": 0.007712, "lr": 0.00042530919606812215, "best_val_loss": 0.006054}
{"epoch": 29, "train_loss": 0.008717, "val_loss": 0.008382, "lr": 0.0004193203929064353, "best_val_loss": 0.006054}
{"epoch": 30, "train_loss": 0.007853, "val_loss": 0.010683, "lr": 0.00041314644038104216, "best_val_loss": 0.006054}
{"epoch": 31, "train_loss": 0.007779, "val_loss": 0.006897, "lr": 0.00040679408961838426, "best_val_loss": 0.006054}
^C
```

## Train Dino Outputs

```SH
$ python main.py train-dino --data-root /mnt/bigdisk/Others/775A2/my_local_dataset_folder/A2_datase
t/Part_A --output-dir /mnt/bigdisk/Others/775A2/my_local_dataset_folder/A2_Outputs/dino
/home/explanet/.conda/envs/netex/lib/python3.10/site-packages/torch/nn/utils/weight_norm.py:143: FutureWarning: `torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.
  WeightNorm.apply(module, name, dim)
{"epoch": 0, "train_loss": 8.290956, "lr": 0.00010006397952655151}
{"epoch": 1, "train_loss": 7.94726, "lr": 0.0002000639795265515}
{"epoch": 2, "train_loss": 7.457182, "lr": 0.0003000639795265515}
{"epoch": 3, "train_loss": 6.67736, "lr": 0.0004000639795265515}
{"epoch": 4, "train_loss": 6.068023, "lr": 0.0005}
{"epoch": 5, "train_loss": 5.703491, "lr": 0.0004998633143352315}
{"epoch": 6, "train_loss": 5.520328, "lr": 0.0004994534068046936}
{"epoch": 7, "train_loss": 5.413679, "lr": 0.0004987707256362529}
{"epoch": 8, "train_loss": 5.198572, "lr": 0.0004978160173317438}
{"epoch": 9, "train_loss": 4.874724, "lr": 0.0004965903258506806}
{"epoch": 10, "train_loss": 4.314158, "lr": 0.0004950949914687023}
{"epoch": 11, "train_loss": 3.828838, "lr": 0.0004933316493120015}
{"epoch": 12, "train_loss": 3.485844, "lr": 0.0004913022275693372}
{"epoch": 13, "train_loss": 3.261388, "lr": 0.0004890089453835894}
{"epoch": 14, "train_loss": 3.390526, "lr": 0.00048645431042515866}
{"epoch": 15, "train_loss": 3.662882, "lr": 0.0004836411161498652}
{"epoch": 16, "train_loss": 3.420222, "lr": 0.0004805724387443462}
{"epoch": 17, "train_loss": 3.114232, "lr": 0.00047725163376229063}
{"epoch": 18, "train_loss": 2.932486, "lr": 0.0004736823324551909}
{"epoch": 19, "train_loss": 2.839219, "lr": 0.00046986843780162223}
{"epoch": 20, "train_loss": 2.769371, "lr": 0.0004658141202393935}

```

## Evaluate Output

```SH
$ python main.py run-all     --probe-data-root /mnt/bigdisk/Others/775A2/my_local_dataset_folder/A2_dataset/Part_Aa     --data-root       /mnt/bigdisk/Others/775A2/my_local_dataset_folder/A2_dataset/Part_A      --clip-ckpt  /mnt/bigdisk/Others/775A2/my_local_dataset_folder/A2_Outputs/clip/checkpoints/best.pt     --dino-ckpt  /mnt/bigdisk/Others/775A2/my_local_dataset_folder/A2_Outputs/dino/checkpoints/last.pt     --tokenizer  /mnt/bigdisk/Others/775A2/my_local_dataset_folder/A2_Outputs/clip/tokenizer.json     --output-dir /mnt/bigdisk/Others/775A2/my_local_dataset_folder/A2_Outputs

############################################################
#  STEP 1/3  -  Linear Probing  (Table 1)
############################################################

[probe]===CLIP===
  task=count
    [cache] clip_count_train.npz
    [cache] clip_count_val.npz
    [cls] train acc=0.3896  val acc=0.3885
    [gap] train acc=0.3774  val acc=0.3736
  task=colors
    [cache] clip_colors_train.npz
    [cache] clip_colors_val.npz
    [cls] train F1=0.9917  val F1=0.9915
    [gap] train F1=0.9620  val F1=0.9619

[probe]===DINO Student===
  task=count
    [cache] dino_student_count_train.npz
    [cache] dino_student_count_val.npz
    [cls] train acc=0.4036  val acc=0.3881
    [gap] train acc=0.3899  val acc=0.3817
  task=colors
    [cache] dino_student_colors_train.npz
    [cache] dino_student_colors_val.npz
    [cls] train F1=0.8097  val F1=0.8101
    [gap] train F1=0.8165  val F1=0.8161

[probe]===DINO Teacher===
  task=count
    [cache] dino_teacher_count_train.npz
    [cache] dino_teacher_count_val.npz
    [cls] train acc=0.3966  val acc=0.3807
    [gap] train acc=0.3901  val acc=0.3788
  task=colors
    [cache] dino_teacher_colors_train.npz
    [cache] dino_teacher_colors_val.npz
    [cls] train F1=0.8019  val F1=0.8022
    [gap] train F1=0.8149  val F1=0.8155

=====================================================================================================================
TABLE 1 - Representation Analysis
=====================================================================================================================

Model                   --- [CLS] Counting ---  -- [CLS] Color Pred. --  --- GAP Counting ---  -- GAP Color Pred. --
                           Train      Val     Train      Val     Train      Val     Train      Val
---------------------------------------------------------------------------------------------------------------------
CLIP                      0.3896   0.3885    0.9917   0.9915    0.3774   0.3736    0.9620   0.9619
DINO Student              0.4036   0.3881    0.8097   0.8101    0.3899   0.3817    0.8165   0.8161
DINO Teacher              0.3966   0.3807    0.8019   0.8022    0.3901   0.3788    0.8149   0.8155
=====================================================================================================================

[probe] Full results -> /mnt/bigdisk/Others/775A2/my_local_dataset_folder/A2_Outputs/probe/table1.json

############################################################
#  STEP 2/3  -  t-SNE Visualisation
############################################################
[tsne] CLIP Vision Encoder:loading cached features from probe step
[tsne] CLIP Vision Encoder:running t-SNE (N=70000,subsample=10000) ...
t-SNE plot saved to /mnt/bigdisk/Others/775A2/my_local_dataset_folder/A2_Outputs/tsne/tsne_clip.pdf
[tsne] DINO Student:loading cached features from probe step
[tsne] DINO Student:running t-SNE (N=70000,subsample=10000) ...
t-SNE plot saved to /mnt/bigdisk/Others/775A2/my_local_dataset_folder/A2_Outputs/tsne/tsne_dino_student.pdf
[tsne] DINO Teacher:loading cached features from probe step
[tsne] DINO Teacher:running t-SNE (N=70000,subsample=10000) ...
t-SNE plot saved to /mnt/bigdisk/Others/775A2/my_local_dataset_folder/A2_Outputs/tsne/tsne_dino_teacher.pdf

############################################################
#  STEP 3/3  -  Cross-Modal Retrieval
############################################################
[retrieval] Encoding validation set ...
{
  "image_to_text_r1": 0.8756,
  "image_to_text_r3": 0.9801333333333333,
  "text_to_image_r1": 0.8684,
  "text_to_image_r3": 0.9781333333333333
}

[retrieval] ----------------------------------------
  Image -> Text   R@1=0.8756
  Image -> Text   R@3=0.9801
  Text  -> Image  R@1=0.8684
  Text  -> Image  R@3=0.9781
[retrieval] Saved to /mnt/bigdisk/Others/775A2/my_local_dataset_folder/A2_Outputs/retrieval

[run-all] All evaluation steps complete.
```