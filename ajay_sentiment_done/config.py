import torch

# Training Settings
BATCH_SIZE = 8  # Lower to 4 if you get "Out of Memory" errors
MAX_LEN = 512   # Maximum words the model reads per article
EPOCHS = 3      # Number of times to study the entire dataset
LEARNING_RATE = 2e-5

# Paths
MODEL_NAME = "microsoft/deberta-v3-base"
SAVE_PATH = "model_v1.pth"

# Device (Auto-detect GPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️  Running on: {DEVICE}")