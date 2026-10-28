"""Download the public resume datasets named in the proposal.

Datasets are downloaded into data/raw/<dataset>/ on the user's machine. The repository itself ships
only the public Resume Atlas copy (MIT licence, see data/raw/resume-atlas/SOURCE.md), so `fetch-data`
finds it already there and downloads nothing; other datasets are never committed.
"""

from __future__ import annotations

import subprocess
import sys
import urllib.request
from pathlib import Path

# Everything needed to download and read each dataset.
DATASETS = {
    "resume-atlas": {
        "source": "huggingface",
        "repo": "ahmedheakl/resume-atlas",
        "files": ["train.csv"],
        "csv": "train.csv",
        "text_col": "Text",
        "label_col": "Category",
        "licence": "MIT (per the dataset card on Hugging Face)",
        "note": "13,389 resumes, 43 categories; text is already lower-cased and punctuation-stripped.",
    },
    "kaggle-resume-dataset": {
        "source": "kaggle",
        "slug": "gauravduttakiit/resume-dataset",
        "csv": "UpdatedResumeDataSet.csv",
        "text_col": "Resume",
        "label_col": "Category",
        "licence": "Check the dataset page on Kaggle before redistributing.",
        "note": "Needs your own Kaggle API token (https://www.kaggle.com/settings).",
    },
}


def dataset_dir(name: str, root: str | Path = "data/raw") -> Path:
    """Where a dataset lives on disk, e.g. data/raw/resume-atlas."""
    return Path(root) / name


def dataset_csv(name: str, root: str | Path = "data/raw") -> Path:
    return dataset_dir(name, root) / DATASETS[name]["csv"]


def fetch_dataset(name: str, root: str | Path = "data/raw") -> Path:
    """Download a known dataset (skips files that already exist). Returns its folder."""
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Known: {sorted(DATASETS)}")
    spec = DATASETS[name]
    out = dataset_dir(name, root)
    out.mkdir(parents=True, exist_ok=True)

    if spec["source"] == "huggingface":
        for filename in spec["files"]:
            target = out / filename
            if target.exists():
                continue
            url = f"https://huggingface.co/datasets/{spec['repo']}/resolve/main/{filename}"
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, target)
    elif spec["source"] == "kaggle":
        if (out / spec["csv"]).exists():
            return out
        # The official Kaggle CLI reads the user's own token from ~/.kaggle.
        subprocess.run(
            [sys.executable, "-m", "kaggle", "datasets", "download", "-d", spec["slug"], "-p", str(out), "--unzip"],
            check=True,
        )
    return out
