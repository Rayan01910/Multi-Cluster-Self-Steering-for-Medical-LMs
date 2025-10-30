"""Convenience wrapper for launching the official RENT entropy baseline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

try:
    from hydra import compose, initialize_config_dir
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "Hydra is required to run the RENT baseline. Install it with 'pip install hydra-core'."
    ) from exc

try:
    from omegaconf import OmegaConf
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "OmegaConf is required to run the RENT baseline. Install it with 'pip install omegaconf'."
    ) from exc

from rent.verl.trainer.main_ppo import run_ppo


@dataclass(frozen=True)
class RentDatasetConfig:
    """Description of a dataset configuration shipped with RENT."""

    exp_name: str
    train_filename: str
    val_filename: str


_DATASETS: dict[str, RentDatasetConfig] = {
    "gsm8k": RentDatasetConfig("gsm8k", "train.parquet", "test.parquet"),
    "math500": RentDatasetConfig("math500", "test.parquet", "test.parquet"),
    "amc": RentDatasetConfig("amc", "test.parquet", "test.parquet"),
    "aime": RentDatasetConfig("aime", "test.parquet", "test.parquet"),
    "gpqa": RentDatasetConfig("gpqa", "test.parquet", "test.parquet"),
    "countdown": RentDatasetConfig("countdown", "train.parquet", "test.parquet"),
    "medqa": RentDatasetConfig("medqa", "train.parquet", "validation.parquet"),
}

_DEFAULT_EXPERIMENTS: Sequence[str] = ("grpo", "entropy", "format", "sampleval")

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "rent" / "verl" / "trainer" / "config"


def _build_logger_override(loggers: Sequence[str]) -> str:
    joined = ", ".join(loggers)
    return f"trainer.logger=[{joined}]"


def _expand_path(path: Path) -> str:
    return str(path.expanduser().resolve())


def _validate_paths(train_path: Path, val_path: Path) -> None:
    missing: list[Path] = [p for p in (train_path, val_path) if not p.exists()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Expected dataset parquet files to exist: {missing_str}")


def _dataset_config(name: str) -> RentDatasetConfig:
    key = name.lower()
    if key not in _DATASETS:
        known = ", ".join(sorted(_DATASETS))
        raise ValueError(f"Unknown dataset '{name}'. Available options: {known}")
    return _DATASETS[key]


def _build_experiment_override(dataset_exp: str, extra_exps: Sequence[str]) -> str:
    experiments: List[str] = list(_DEFAULT_EXPERIMENTS)
    experiments.append(dataset_exp)
    experiments.extend(extra_exps)
    joined = ", ".join(experiments)
    return f"exps=[{joined}]"


def run_rent_baseline(
    dataset: str,
    base_model: str,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    num_gpus: int | None = None,
    project_name: str = "rent_baseline",
    experiment_name: str | None = None,
    loggers: Sequence[str] = ("console",),
    extra_experiments: Sequence[str] = (),
    overrides: Iterable[str] = (),
) -> None:
    """Launch the RENT entropy-minimisation training loop.

    This wrapper mirrors the command line invocation from the RENT repository::

        python -m rent.verl.trainer.main_ppo \\
            exps="[grpo, entropy, format, sampleval, <dataset>]" \\
            base_model=<hf repo id>

    Args:
        dataset: One of the datasets packaged with RENT (``gsm8k``, ``math500``,
            ``amc``, ``aime``, ``gpqa`` or ``countdown``).
        base_model: HuggingFace model identifier or local path for the policy.
        data_dir: Directory containing the parquet files produced by RENT's
            preprocessing scripts.  Defaults to ``~/data/<dataset>``.
        output_dir: Optional location for checkpoints/logs.  Defaults to the
            RENT convention ``checkpoints/<project>/<experiment>``.
        num_gpus: Overrides the ``ngpus`` config entry.
        project_name: Value for ``trainer.project_name``.
        experiment_name: Value for ``trainer.experiment_name``.  Defaults to the
            dataset name when omitted.
        loggers: Logger backends to enable.  ``["console"]`` disables WandB.
        extra_experiments: Additional Hydra experiment configs to append to the
            ``exps`` override.
        overrides: Raw Hydra override strings for advanced customisation.
    """

    cfg = _dataset_config(dataset)
    default_data_root = Path.home() / "data" / cfg.exp_name
    data_root = data_dir or default_data_root
    train_path = data_root / cfg.train_filename
    val_path = data_root / cfg.val_filename
    _validate_paths(train_path, val_path)

    exp_override = _build_experiment_override(cfg.exp_name, extra_experiments)
    logger_override = _build_logger_override(loggers)

    hydra_overrides: List[str] = [
        exp_override,
        f"base_model={base_model}",
        f"data.train_files={_expand_path(train_path)}",
        f"data.val_files={_expand_path(val_path)}",
        f"trainer.project_name={project_name}",
        f"trainer.experiment_name={experiment_name or cfg.exp_name}",
        logger_override,
    ]
    if num_gpus is not None:
        hydra_overrides.append(f"ngpus={num_gpus}")
    if output_dir is not None:
        hydra_overrides.append(f"trainer.default_local_dir={_expand_path(output_dir)}")
    hydra_overrides.extend(list(overrides))

    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=hydra_overrides)
        OmegaConf.resolve(config)
    run_ppo(config)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the official RENT baseline")
    parser.add_argument(
        "dataset",
        choices=sorted(_DATASETS),
        help="Dataset name matching one of RENT's pre-packaged experiment configs.",
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="HuggingFace model repo or local path to fine-tune with RENT.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Directory containing train.parquet/test.parquet produced by RENT preprocessors.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory for checkpoints/logs (trainer.default_local_dir).",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        help="Override the number of GPUs passed to RENT (ngpus).",
    )
    parser.add_argument(
        "--project-name",
        default="rent_baseline",
        help="Override trainer.project_name.",
    )
    parser.add_argument(
        "--experiment-name",
        help="Override trainer.experiment_name (defaults to the dataset name).",
    )
    parser.add_argument(
        "--logger",
        nargs="+",
        default=["console"],
        help="Logger backends to enable (default: console only).",
    )
    parser.add_argument(
        "--extra-experiment",
        dest="extra_experiments",
        action="append",
        default=[],
        help="Additional Hydra experiment configs to append to the exps list.",
    )
    parser.add_argument(
        "--override",
        dest="overrides",
        action="append",
        default=[],
        help="Arbitrary Hydra override strings for advanced customisation.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv or sys.argv[1:])
    run_rent_baseline(
        dataset=args.dataset,
        base_model=args.base_model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_gpus=args.num_gpus,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        loggers=args.logger,
        extra_experiments=args.extra_experiments,
        overrides=args.overrides,
    )


if __name__ == "__main__":
    main()
