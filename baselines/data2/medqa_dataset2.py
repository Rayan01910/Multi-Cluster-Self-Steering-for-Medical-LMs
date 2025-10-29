from datasets import load_dataset
from torch.utils.data import Dataset

LETTER = ["A","B","C","D"]

class MedQADataset(Dataset):
    """
    Exposes items with:
      - stem: main question (sent1)
      - choices: list[str] length 4 (ending0..3)
      - label: int in {0,1,2,3}
      - qid: str id
    """
    def __init__(self, split="train", hf_name="GBaker/MedQA-USMLE-4-options-hf"):
        ds = load_dataset(hf_name, split=split)

        cols = set(ds.column_names)
        # Original ``medqa`` conversion hosted at GBaker exposes the four answer
        # options as ``ending0``..``ending3`` alongside the ``sent1`` stem. Some
        # mirrors on the Hub expose HuggingFace's default MedQA schema instead
        # where the question lives under ``question`` and the answer choices are
        # bundled inside an ``options`` column.  Support both layouts so that the
        # downstream calibration code can run regardless of which variant the
        # user has cached locally.

        if {"sent1", "ending0", "ending1", "ending2", "ending3", "label"}.issubset(cols):
            self.ids = ds["id"]
            self.stems = ds["sent1"]
            self.choices = list(
                zip(ds["ending0"], ds["ending1"], ds["ending2"], ds["ending3"])  # type: ignore[arg-type]
            )
            self.labels = ds["label"]
        elif {"question", "options"}.issubset(cols):
            self.ids = ds["id"] if "id" in cols else list(range(len(ds)))
            self.stems = ds["question"]

            processed_choices = []
            for raw in ds["options"]:
                # ``raw`` can either be a list ordered from A..D or a dict keyed
                # by option letter / index. Normalise into a tuple ordered by LETTER.
                if isinstance(raw, dict):
                    choices = []
                    for k, fallback in zip(LETTER, range(len(LETTER))):
                        if k in raw:
                            choices.append(raw[k])
                        elif str(fallback) in raw:
                            choices.append(raw[str(fallback)])
                        else:
                            raise KeyError(f"Missing option '{k}' in MedQA example")
                    processed_choices.append(tuple(choices))
                else:
                    processed_choices.append(tuple(raw))
            self.choices = processed_choices

            if "label" in cols:
                self.labels = ds["label"]
            elif "answer_idx" in cols:
                self.labels = ds["answer_idx"]
            elif "answer" in cols:
                answers = ds["answer"]
                self.labels = [LETTER.index(ans) if isinstance(ans, str) else int(ans) for ans in answers]
            else:
                raise ValueError("Unsupported MedQA schema: could not locate labels")
        else:
            raise ValueError("Unsupported MedQA schema: expected either ending0..3 or options column")

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        return dict(
            qid=self.ids[i],
            stem=self.stems[i],
            choices=list(self.choices[i]),
            label=int(self.labels[i]),
        )
