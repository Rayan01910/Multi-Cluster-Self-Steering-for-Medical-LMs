from datasets import load_dataset

def load_medqa(split):
    ds = load_dataset("GBaker/MedQA-USMLE-4-options")
    if split == "test":
        return ds["test"]
    elif split == "train":
        return ds["train"]
    elif split == "validation":
        # Create a validation split locally by splitting 10% from train
        train_ds = ds["train"]
        val_split = train_ds.train_test_split(test_size=0.1, seed=42)
        return val_split["test"]
    else:
        return ds
