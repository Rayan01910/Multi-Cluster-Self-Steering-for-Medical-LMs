"""Utilities for exporting MedQA into RENT's parquet format."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from datasets import Dataset

from baselines.data2.medqa_dataset2 import LETTER, MedQADataset
from baselines.data2.prompt_builder2 import build_prompt


SYSTEM_PROMPT = "You are a helpful medical reasoning assistant."
DEFAULT_DATA_SOURCE = "GBaker/MedQA-USMLE-4-options-hf"


def _build_messages(prompt: str, *, include_system: bool) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if include_system:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": prompt})
    return messages


@dataclass
class ExportConfig:
    output_dir: Path
    hf_name: str
    train_split: str
    val_split: str
    include_system_prompt: bool
    limit: int | None


def _iter_examples(dataset: MedQADataset, *, include_system_prompt: bool, data_source: str) -> Iterable[dict]:
    for record in dataset:
        prompt = build_prompt(record["stem"], record["choices"])
        label = LETTER[record["label"]]

        yield {
            "prompt": _build_messages(prompt, include_system=include_system_prompt),
            "data_source": data_source,
            "ability": "medical",
            "metadata": {
                "question_id": record["qid"],
                "choices": record["choices"],
            },
            "reward_model": {
                "style": "multiple_choice",
                "ground_truth": label,
            },
        }


def _write_parquet(examples: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(list(examples)).to_parquet(str(path))


def export_medqa(config: ExportConfig) -> None:
    data_source = config.hf_name or DEFAULT_DATA_SOURCE

    train_dataset = MedQADataset(split=config.train_split, hf_name=data_source)
    val_dataset = MedQADataset(split=config.val_split, hf_name=data_source)

    if config.limit is not None:
        train_examples = list(_iter_examples(train_dataset, include_system_prompt=config.include_system_prompt, data_source=data_source))[: config.limit]
        val_examples = list(_iter_examples(val_dataset, include_system_prompt=config.include_system_prompt, data_source=data_source))[: config.limit]
    else:
        train_examples = list(_iter_examples(train_dataset, include_system_prompt=config.include_system_prompt, data_source=data_source))
        val_examples = list(_iter_examples(val_dataset, include_system_prompt=config.include_system_prompt, data_source=data_source))

    _write_parquet(train_examples, config.output_dir / "train.parquet")
    _write_parquet(val_examples, config.output_dir / "validation.parquet")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MedQA splits into RENT-ready parquet files")
    parser.add_argument("output_dir", type=Path, help="Destination directory for train/validation parquet files")
    parser.add_argument("--hf-name", default=DEFAULT_DATA_SOURCE, help="HuggingFace dataset identifier (default: %(default)s)")
    parser.add_argument("--train-split", default="train", help="Split name to use for training data (default: %(default)s)")
    parser.add_argument(
        "--val-split",
        default="validation",
        help="Split name to use for validation/evaluation data (default: %(default)s)",
    )
    parser.add_argument(
        "--include-system-prompt",
        action="store_true",
        help="Prepend a system message describing the assistant role.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional cap on the number of examples written to each split (useful for smoke tests).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    export_medqa(
        ExportConfig(
            output_dir=args.output_dir,
            hf_name=args.hf_name,
            train_split=args.train_split,
            val_split=args.val_split,
            include_system_prompt=args.include_system_prompt,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()
