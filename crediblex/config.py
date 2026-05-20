# --- Model ---
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LEN = 96              # token limit per window (VRAM constraint reduced from 128)
STRIDE = 48               # sliding window stride
MAX_WINDOWS = 5           # hard cap on windows per article (OOM guard)

# --- Device ---
import torch
DEVICE = torch.device("cpu")

# --- Paths ---
SAVE_PATH = "model_v1.pth"
LOG_PATH   = "logs/requests.jsonl"

# --- Training (VRAM safe-mode) ---
BATCH_SIZE  = 1
GRAD_ACCUM  = 32          # effective batch = 32
EPOCHS      = 5
LR          = 2e-5
WARMUP_RATIO = 0.1
FREEZE_LAYERS = 9         # freeze more layers (9 of 12) to reduce optimizer VRAM footprint

# --- Head output sizes (source of truth for ALL files) ---
NUM_BIAS_LABELS      = 5   # 0=slightly_left 1=left 2=center 3=right 4=slightly_right
NUM_FACT_LABELS      = 1   # regression scalar, float 0.0–1.0
NUM_INTENT_LABELS    = 3   # 0=news 1=opinion 2=satire
NUM_EMOTION_LABELS   = 28  # GoEmotions 28-class multi-label

# --- Bias class weights (tensor, for CrossEntropyLoss) ---
# Extreme classes (0,4) are minority — upweight to fix 16:1 imbalance
import torch as _t
BIAS_CLASS_WEIGHTS = _t.tensor([2.5, 1.2, 0.7, 1.2, 2.5], dtype=torch.float32)
