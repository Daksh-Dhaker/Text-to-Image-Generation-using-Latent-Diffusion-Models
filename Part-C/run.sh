source ~/data1/miniconda3/etc/profile.d/conda.sh
conda activate my_env

set -eu  


if [ $# -lt 1 ]; then
    echo "Usage: $0 /path/to/dataset/circuits/A"
    exit 1
fi

DATA_ROOT="$1"

if [ ! -d "$DATA_ROOT" ]; then
    echo "[ERROR] Data directory not found: $DATA_ROOT"
    exit 1
fi



SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_DIR="${CKPT_DIR:-${SCRIPT_DIR}/checkpoints}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"

# VAE hyperparameters
VAE_EPOCHS="${VAE_EPOCHS:-150}"
VAE_BATCH="${VAE_BATCH:-64}"
VAE_LR="${VAE_LR:-2e-4}"
VAE_KL_WEIGHT="${VAE_KL_WEIGHT:-1e-6}"
VAE_WORKERS="${VAE_WORKERS:-4}"
VAE_FID_EVERY="${VAE_FID_EVERY:-10}"   

# LDM hyperparameters
LDM_EPOCHS="${LDM_EPOCHS:-150}"
LDM_BATCH="${LDM_BATCH:-64}"
LDM_LR="${LDM_LR:-1e-4}"
LDM_TIMESTEPS="${LDM_TIMESTEPS:-500}"
LDM_GUIDANCE="${LDM_GUIDANCE:-4.0}"
LDM_WORKERS="${LDM_WORKERS:-4}"
LDM_FID_EVERY="${LDM_FID_EVERY:-10}"  
LDM_FID_SAMPLES="${LDM_FID_SAMPLES:-1000}" 


EVAL_FID_SAMPLES="${EVAL_FID_SAMPLES:-10000}"



VAE_CKPT_DIR="${CKPT_DIR}/vae"
LDM_CKPT_DIR="${CKPT_DIR}/ldm"
mkdir -p "$VAE_CKPT_DIR" "$LDM_CKPT_DIR" "$LOG_DIR"

VAE_LOG="${LOG_DIR}/vae_train.log"
LDM_LOG="${LOG_DIR}/ldm_train.log"
EVAL_VAE_LOG="${LOG_DIR}/eval_vae.log"
EVAL_LDM_LOG="${LOG_DIR}/eval_ldm.log"
VAE_FID_LOG="${VAE_CKPT_DIR}/fid_log.jsonl"


log "Stage 1/2 — Training VAE  (${VAE_EPOCHS} epochs)"
echo "  Data:       $DATA_ROOT"
echo "  Checkpoint: $VAE_CKPT_DIR"
echo "  Log:        $VAE_LOG"
echo "  FID every:  ${VAE_FID_EVERY} epochs"
echo ""

cd "$SCRIPT_DIR"

python3 ~/misc/internet/proxyiit.py
python3 train_vae.py \
    --data_root   "$DATA_ROOT"    \
    --save_dir    "$VAE_CKPT_DIR" \
    --epochs      "$VAE_EPOCHS"   \
    --batch_size  "$VAE_BATCH"    \
    --lr          "$VAE_LR"       \
    --kl_weight   "$VAE_KL_WEIGHT"\
    --workers     "$VAE_WORKERS"  \
    --fid_every   "$VAE_FID_EVERY" \
    2>&1 | tee "$VAE_LOG"


VAE_BEST="${VAE_CKPT_DIR}/vae_best.pt"
if [ ! -f "$VAE_BEST" ]; then
    err "VAE best checkpoint not found at $VAE_BEST — training may have failed."
    exit 1
fi
ok "VAE training complete. Best checkpoint: $VAE_BEST"


VAE_FID_LOG="${VAE_CKPT_DIR}/fid_log.jsonl"
if [ -f "$VAE_FID_LOG" ]; then
    echo ""
    echo "  VAE Reconstruction FID Progress:"
    python - "$VAE_FID_LOG" <<'EOF'
import sys, json
with open(sys.argv[1]) as f:
    rows = [json.loads(l) for l in f if l.strip()]
for r in rows:
    bar_len = max(1, int(r['fid'] / 5))
    bar = '█' * min(bar_len, 40)
    print(f"    Epoch {r['epoch']:4d}  FID={r['fid']:7.2f}  {bar}")
EOF
fi

log "Stage 2/2 — Training LDM  (${LDM_EPOCHS} epochs)"
echo "  VAE ckpt:   $VAE_BEST"
echo "  Checkpoint: $LDM_CKPT_DIR"
echo "  Log:        $LDM_LOG"
echo "  FID every:  ${LDM_FID_EVERY} epochs  (n=${LDM_FID_SAMPLES} samples)"
echo ""
python3 ~/misc/internet/proxyiit.py
python3 train_ldm.py \
    --data_root      "$DATA_ROOT"      \
    --vae_ckpt       "$VAE_BEST"       \
    --save_dir       "$LDM_CKPT_DIR"   \
    --epochs         "$LDM_EPOCHS"     \
    --batch_size     "$LDM_BATCH"      \
    --lr             "$LDM_LR"         \
    --timesteps      "$LDM_TIMESTEPS"  \
    --guidance_scale "$LDM_GUIDANCE"   \
    --workers        "$LDM_WORKERS"    \
    --fid_every      "$LDM_FID_EVERY"  \
    --fid_samples    "$LDM_FID_SAMPLES"\
    --cache_latents                    \
    2>&1 | tee "$LDM_LOG"

LDM_BEST="${LDM_CKPT_DIR}/ldm_best.pt"
if [ ! -f "$LDM_BEST" ]; then
    err "LDM best checkpoint not found at $LDM_BEST — training may have failed."
    exit 1
fi
ok "LDM training complete. Best checkpoint: $LDM_BEST"


LDM_FID_LOG="${LDM_CKPT_DIR}/fid_log.jsonl"
if [ -f "$LDM_FID_LOG" ]; then
    echo ""
    echo "  LDM Generation FID Progress:"
    python - "$LDM_FID_LOG" <<'EOF'
import sys, json
with open(sys.argv[1]) as f:
    rows = [json.loads(l) for l in f if l.strip()]
for r in rows:
    bar_len = max(1, int(r['fid'] / 10))
    bar = '█' * min(bar_len, 40)
    print(f"    Epoch {r['epoch']:4d}  FID={r['fid']:7.2f}  {bar}")
EOF
fi

log "Final evaluation — computing full-dataset FID …"

LATENT_STATS="${LDM_CKPT_DIR}/latent_stats.pt"

echo ""
echo "  ── VAE Reconstruction FID ──"
python3 ~/misc/internet/proxyiit.py
python3 evaluate.py vae \
    --data_root  "$DATA_ROOT"         \
    --vae_ckpt   "$VAE_BEST"          \
    --output_dir "${CKPT_DIR}/eval/vae" \
    --batch_size 64                   \
    2>&1 | tee "$EVAL_VAE_LOG"

echo ""
echo "  ── LDM Generation FID ──"
python3 ~/misc/internet/proxyiit.py
python3 evaluate.py ldm \
    --data_root      "$DATA_ROOT"          \
    --vae_ckpt       "$VAE_BEST"           \
    --ldm_ckpt       "$LDM_BEST"           \
    --latent_stats   "$LATENT_STATS"       \
    --output_dir     "${CKPT_DIR}/eval/ldm" \
    --batch_size     16                    \
    --timesteps      "$LDM_TIMESTEPS"      \
    --guidance_scale "$LDM_GUIDANCE"       \
    --sampler        ddpm                  \
    --sample_steps   "$LDM_TIMESTEPS"      \
    2>&1 | tee "$EVAL_LDM_LOG"

log "All done! Summary:"
echo ""
echo "  Checkpoints"
echo "    VAE (best loss) : $VAE_BEST"
echo "    LDM (best loss) : $LDM_BEST"
[ -f "${VAE_CKPT_DIR}/vae_best_fid.pt" ] && \
    echo "    VAE (best FID)  : ${VAE_CKPT_DIR}/vae_best_fid.pt"
[ -f "${LDM_CKPT_DIR}/ldm_best_fid.pt" ] && \
    echo "    LDM (best FID)  : ${LDM_CKPT_DIR}/ldm_best_fid.pt"
echo ""
echo "  FID logs  (JSONL)"
[ -f "$VAE_FID_LOG" ] && echo "    VAE : $VAE_FID_LOG"
[ -f "$LDM_FID_LOG" ] && echo "    LDM : $LDM_FID_LOG"
echo ""
echo "  Training logs"
echo "    VAE : $VAE_LOG"
echo "    LDM : $LDM_LOG"
echo ""
echo "  Visualizations"
echo "    VAE reconstructions : ${VAE_CKPT_DIR}/reconstructions/"
echo "    LDM samples         : ${LDM_CKPT_DIR}/generated_viz/"
echo "    Final eval images   : ${CKPT_DIR}/eval/"
echo ""
ok "Pipeline complete."
