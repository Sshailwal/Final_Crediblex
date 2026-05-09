import torch
import os

# ─────────────────────────────────────────────────────────────────────────────
# Training Hyperparameters - BALANCED MODE (RTX 4050 6GB)
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE        = 2       # 2 articles at a time — better GPU utilization
MAX_LEN           = 192     # 192 tokens — more context than safe mode
GRAD_ACCUM_STEPS  = 16      # effective batch = 2×16 = 32 (consistent with safe mode)
EPOCHS            = 3       # Fine-tune run from epoch_5.pth — emotion pos_weight + HuberLoss
LEARNING_RATE     = 5e-5
WARMUP_RATIO      = 0.10
SLIDING_STRIDE    = 96      # stride for sliding window (= MAX_LEN // 2)
FREEZE_LAYERS     = 4       # freeze bottom 4 layers (was 6) — unfreezing 2 more

DROPOUT           = 0.1
GRAD_CLIP         = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "microsoft/deberta-v3-base"

# Memory Optimizations
GRADIENT_CHECKPOINTING = True   # keeps activation memory low
USE_8BIT_ADAM          = False  # set True if bitsandbytes is installed

N_BIAS_CLASSES    = 3   # Left=0, Center=1, Right=2 (3-class balanced scheme)
N_EMOTION_CLASSES = 7   # 7 Ekman classes

# ─────────────────────────────────────────────────────────────────────────────
# Multi-Task Loss Weights
# ─────────────────────────────────────────────────────────────────────────────
BIAS_LOSS_WEIGHT    = 2.0
FACT_LOSS_WEIGHT    = 1.0     # note: user specified 1.0 in prompt, was 10.0 before
EMOTION_LOSS_WEIGHT = 1.5

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint / Paths
# ─────────────────────────────────────────────────────────────────────────────
MID_EPOCH_CKPT_FREQ = 500
SAVE_PATH           = "rediblex_v1.pth"
CHECKPOINT_DIR      = os.path.join(os.environ.get("TEMP", "C:\\Temp"), "crediblex_checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Safety & Device
# ─────────────────────────────────────────────────────────────────────────────
VRAM_SAFETY_THRESHOLD_GB = 5.0  # Tighter threshold for Ollama coexistence
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    print("Balanced Mode Active | Device: {}".format(DEVICE))
    if DEVICE.type == "cuda":
        print("GPU   : {}".format(torch.cuda.get_device_name(0)))
        print("VRAM  : {:.1f} GB total".format(torch.cuda.get_device_properties(0).total_memory / 1e9))