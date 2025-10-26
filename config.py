from dataclasses import dataclass

@dataclass
class Config:
    # Data
    hf_dataset: str = "GBaker/MedQA-USMLE-4-options-hf"
    text_field: str = "sent1"
    option_fields: tuple[str, str, str, str] = ("ending0", "ending1", "ending2", "ending3")
    label_field: str = "label"
    train_split: str = "train"

    # Model
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"  # change if needed
    device: str = "cuda"
    fp16: bool = True
    max_length: int = 1024
    penultimate_layer_offset: int = 2  # take hidden states from last_hidden_state of layer - 2

    # Steering / clustering
    num_clusters: int = 32
    kmeans_seed: int = 42
    cosine_temp: float = 15.0  # τ for softmax over cosine sims
    alpha: float = 0.25        # steering strength

    # Projection training
    proj_lr: float = 1e-3
    proj_epochs: int = 2
    proj_batch_size: int = 64
    proj_weight_decay: float = 1e-4

    # Calibrator
    calib_val_frac: float = 0.2  # split from train if needed

    # Evaluation
    ece_bins: int = 10

cfg = Config()
