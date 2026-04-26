import torch
import os

# ─────────────────────────────────────────────────────────────────────────────
# Training Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE        = 1     # Safe Mode: BATCH_SIZE=1 for RTX 4050 6GB with Sliding Windows
MAX_LEN           = 256   # 256 tokens
EPOCHS            = 7
LEARNING_RATE     = 2e-5
GRAD_ACCUM_STEPS  = 32    # Effective batch = 1 x 32 = 32

WARMUP_RATIO      = 0.06  # 6% of total steps for linear LR warmup
DROPOUT           = 0.1
GRAD_CLIP         = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────────────────────────────────────
N_BIAS_CLASSES    = 5     # 0=Far-Left / 1=Slightly-Left / 2=Center / 3=Slightly-Right / 4=Far-Right

# ─────────────────────────────────────────────────────────────────────────────
# Multi-Task Loss Weights
# ─────────────────────────────────────────────────────────────────────────────
BIAS_LOSS_WEIGHT    = 2.0
FACT_LOSS_WEIGHT    = 10.0
EMOTION_LOSS_WEIGHT = 1.5

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint / Crash-Recovery
# ─────────────────────────────────────────────────────────────────────────────
MID_EPOCH_CKPT_FREQ = 500   # Save mid-epoch checkpoint every N batches

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME      = "microsoft/deberta-v3-base"
SAVE_PATH       = "best_bias_acc.pth"
# Checkpoints saved OUTSIDE OneDrive to prevent sync conflicts corrupting .pth files
CHECKPOINT_DIR  = os.path.join(os.environ.get("TEMP", "C:\\Temp"), "crediblex_checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Device (Auto-detect CUDA)
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    _gpu  = torch.cuda.get_device_name(0)
    _vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print("GPU: {}  ({:.1f} GB VRAM)".format(_gpu, _vram))
else:
    print("Running on CPU (no GPU detected)")