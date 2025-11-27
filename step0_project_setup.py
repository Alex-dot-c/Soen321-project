import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List  # <-- for Python < 3.9

import torch
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

# ===== Config (match assignment) =====
TRAIN_PER_CLASS = 500     # per class, total 5,000 train images
TEST_PER_CLASS = 100      # per class, total 1,000 test images
SELECT_MODE = "first"     # "first" or "random"
RANDOM_SEED = 42

DATA_ROOT = "./data"
OUT_TRAIN_DIR = Path("./training_images")
OUT_TEST_DIR = Path("./test_images")

# Saving options:
SAVE_FULL_INDICES = False           # True -> store full indices; False -> manifest only
INDICES_PATH = Path("selected_indices.json")
# ====================================

#Original classes from CIFAR-10
CLASS_NAMES = ["airplane","automobile","bird","cat","deer",
               "dog","frog","horse","ship","truck"]

# Set random seeds for reproducibility
def set_seed(seed: int) -> None:
    """Ensure reproducible sampling."""
    random.seed(seed)
    torch.manual_seed(seed)

# Create output directories
def ensure_dirs() -> None:
    """Create output folders for train/test and each class."""
    OUT_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TEST_DIR.mkdir(parents=True, exist_ok=True)
    for cname in CLASS_NAMES:
        (OUT_TRAIN_DIR / cname).mkdir(parents=True, exist_ok=True)
        (OUT_TEST_DIR / cname).mkdir(parents=True, exist_ok=True)

# Load CIFAR-10 dataset
def load_cifar10():
    """Load CIFAR-10 with raw pixels in [0,1] (no normalization) for PNG export."""
    tf = transforms.ToTensor()
    train_raw = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=tf)
    test_raw  = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=tf)
    return train_raw, test_raw

# Select indices per class
def pick_indices_per_class(dataset, per_class: int, mode: str) -> Dict[int, List[int]]:
    """
    Return {class_id: [indices]} with 'per_class' samples per class.
    mode="first": take first N per class in dataset order.
    mode="random": uniform random sample per class (requires set_seed()).
    """
    # Collect indices by class
    by_class = defaultdict(list)
    if mode == "first":
        for idx, (_, y) in enumerate(dataset):
            if len(by_class[y]) < per_class:
                by_class[y].append(idx)
            if all(len(by_class[c]) >= per_class for c in range(10)):
                break
    elif mode == "random":
        # First gather all indices per class
        # Then sample uniformly
        all_by_class = defaultdict(list)
        for idx, (_, y) in enumerate(dataset):
            all_by_class[y].append(idx)
        for c in range(10):
            if len(all_by_class[c]) < per_class:
                raise ValueError(f"class {c} has {len(all_by_class[c])} samples, < {per_class}")
            by_class[c] = random.sample(all_by_class[c], per_class)
    else:
        raise ValueError("SELECT_MODE must be 'first' or 'random'")
    return by_class

# Save selected images as PNG
def save_images(dataset, indices_by_class: Dict[int, List[int]], out_dir: Path) -> None:
    """Save selected samples as PNG under out_dir/<class_name>/<class>_<k>.png."""
    for c in range(10):
        cdir = out_dir / CLASS_NAMES[c]
        cdir.mkdir(parents=True, exist_ok=True)
        for k, idx in enumerate(indices_by_class[c]):
            img, _ = dataset[idx]               # (C, H, W) in [0,1]
            img = img.permute(1, 2, 0).numpy()  # -> (H, W, C)
            plt.imsave(cdir / f"{CLASS_NAMES[c]}_{k:03d}.png", img)

# Load previously saved indices if they exist
def maybe_load_indices():
    """
    If indices JSON exists:
      - If it contains full indices -> return them.
      - If it is a manifest (no indices) -> return None and let caller recompute.
    """
    if not INDICES_PATH.exists():
        return None, None
    with open(INDICES_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    train_idx = meta.get("train_indices")
    test_idx = meta.get("test_indices")
    return train_idx, test_idx

# Save selected indices or manifest
def save_indices(train_idx, test_idx) -> None:
    """Save either full indices or a minimal manifest, based on SAVE_FULL_INDICES."""
    base_meta = {
        "train_per_class": TRAIN_PER_CLASS,
        "test_per_class": TEST_PER_CLASS,
        "select_mode": SELECT_MODE,
        "random_seed": RANDOM_SEED,
    }
    if SAVE_FULL_INDICES:
        base_meta["train_indices"] = {CLASS_NAMES[c]: train_idx[c] for c in range(10)}
        base_meta["test_indices"]  = {CLASS_NAMES[c]: test_idx[c]  for c in range(10)}
    with open(INDICES_PATH, "w", encoding="utf-8") as f:
        json.dump(base_meta, f, indent=2)
    print(f"Saved indices manifest to {INDICES_PATH}")

# Main
def main() -> None:
    print("== Preparing output directories ==")
    ensure_dirs()

    print("\n== Setting random seed ==")
    set_seed(RANDOM_SEED)

    print("\n== Loading CIFAR-10 ==")
    train_raw, test_raw = load_cifar10()
    print(f"Train size: {len(train_raw)}, Test size: {len(test_raw)}")

    print("\n== Selecting indices per class ==")
    cached_train, cached_test = maybe_load_indices()
    if cached_train is not None and cached_test is not None:
        # If a previous full-indices file exists, reuse it exactly.
        train_idx = {i: cached_train[CLASS_NAMES[i]] for i in range(10)}
        test_idx  = {i: cached_test[CLASS_NAMES[i]]  for i in range(10)}
        print("Loaded full indices from existing JSON.")
    else:
        # Compute indices according to config (manifest or fresh sampling).
        train_idx = pick_indices_per_class(train_raw, TRAIN_PER_CLASS, SELECT_MODE)
        test_idx  = pick_indices_per_class(test_raw,  TEST_PER_CLASS,  SELECT_MODE)
        save_indices(train_idx, test_idx)

    print("\n== Exporting PNGs ==")
    save_images(train_raw, train_idx, OUT_TRAIN_DIR)
    save_images(test_raw,  test_idx,  OUT_TEST_DIR)

    print("\nCompleted.")
    print(f"- Training: {TRAIN_PER_CLASS} per class -> {TRAIN_PER_CLASS * 10} images at {OUT_TRAIN_DIR.resolve()}")
    print(f"- Test:     {TEST_PER_CLASS} per class -> {TEST_PER_CLASS * 10} images at {OUT_TEST_DIR.resolve()}")
    print(f"- Indices JSON: {INDICES_PATH.resolve()}")

if __name__ == "__main__":
    main()
