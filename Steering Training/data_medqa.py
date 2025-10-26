from datasets import load_dataset
from config import cfg

def load_medqa():
    """
    Loads the GBaker/MedQA-USMLE-4-options-hf dataset, which only contains a 'train' split.
    Automatically splits into train/val/test locally for our steering experiments.
    """
    ds = load_dataset(cfg.hf_dataset)
    base_train = ds["train"]

    # 80/10/10 split
    split_1 = base_train.train_test_split(test_size=0.2, seed=cfg.kmeans_seed)
    train_split = split_1["train"]
    temp_split = split_1["test"]

    val_test_split = temp_split.train_test_split(test_size=0.5, seed=cfg.kmeans_seed + 1)
    val_split = val_test_split["train"]
    test_split = val_test_split["test"]

    print(f"Dataset loaded and split → train={len(train_split)} | val={len(val_split)} | test={len(test_split)}")

    return {"train": train_split, "validation": val_split, "test": test_split}
