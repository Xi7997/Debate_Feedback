import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TASKS = {
    "trial": ("Trial Prediction", "case_background", "labels"),
    "article": ("Legal Document Prediction", "fact", "relevant_articles"),
    "crime": ("Crime Prediction", "fact", "accusation"),
}


def dataset_path(task, split):
    folder = ROOT / "Dataset" / TASKS[task][0]
    name = ("eval" if split == "valid" else split) + ".json"
    if task != "trial":
        name = "present_" + split + "_data.json"
    return folder / name


def read_dataset(task, split, path=None):
    text_key, label_key = TASKS[task][1:]
    rows = []
    with open(path or dataset_path(task, split), encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            data = json.loads(line)
            label = data[label_key]
            if isinstance(label, (list, dict)):
                raise ValueError("Each input label must be a single class")
            rows.append({"id": i, "text": data[text_key], "label": str(label)})
    return rows


def load_task(task):
    data = {split: read_dataset(task, split) for split in ("train", "valid", "test")}
    # Exclude validation and test texts from training to prevent data leakage.
    held_out = {row["text"] for split in ("valid", "test") for row in data[split]}
    data["train"] = [row for row in data["train"] if row["text"] not in held_out]
    # Use a fixed task label list; the training subset may not cover every class.
    with (ROOT / "Dataset" / "labels.json").open(encoding="utf-8") as f:
        labels = json.load(f)[task]
    for split in ("valid", "test"):
        unknown = {row["label"] for row in data[split]} - set(labels)
        if unknown:
            raise ValueError(f"{split} contains labels missing from the task label list: {unknown}")
    return data, labels
