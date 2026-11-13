import os

# =========================
# Dataset
# =========================

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # -> src
    "data"
)
DATA_FILE = os.path.join(DATA_DIR, "EITdata.npz")


# ========================
# Model Weights
# ========================

WEIGHTS_DIR = os.path.join(DATA_DIR, "weights")

# ========================
# Checkpoints
# ========================

CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")