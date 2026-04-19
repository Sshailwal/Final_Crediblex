import torch
import os

# ─────────────────────────────────────────────────────────────────────────────
# Training Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE        = 4     # Power-of-2 required to prevent cuBLAS FP16 errors on RTX 40-series
MAX_LEN           = 512   # Max tokens per article/message
EPOCHS            = 5     # ~6-8 hours overnight on RTX 4050
LEARNING_RATE     = 2e-5
GRAD_ACCUM_STEPS  = 6     # Effective batch = 4 × 6 = 24 (better generalisation)

WARMUP_RATIO      = 0.06  # 6% of total steps for linear LR warmup
DROPOUT           = 0.1   # Dropout before all classification heads (prevents overfit)
GRAD_CLIP         = 1.0   # max_norm for gradient clipping (prevents exploding gradients)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME      = "microsoft/deberta-v3-base"
SAVE_PATH       = "model_v1.pth"
# IMPORTANT: checkpoints are saved OUTSIDE OneDrive to prevent OneDrive
# sync conflicts corrupting large .pth files during torch.save() writes.
CHECKPOINT_DIR  = os.path.join(os.environ.get("TEMP", "C:\\Temp"), "crediblex_checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Device (Auto-detect CUDA)
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    _gpu = torch.cuda.get_device_name(0)
    _vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Running on: {DEVICE}  ({_gpu}, {_vram:.1f} GB VRAM)")
else:
    print(f"Running on: {DEVICE}  (no GPU detected)")