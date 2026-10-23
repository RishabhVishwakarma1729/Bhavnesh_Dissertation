"""Pack code + dataset + embedding model into ONE opaque .dat file for an offline Kaggle kernel.

Why one .dat? Kaggle auto-extracts uploaded zips and that step silently fails for many files, so
the zip is renamed to .dat (stored as one file) and unzipped by the kernel itself. Kaggle kernels
may also have no internet, so the embedding model and (optionally) wheels travel inside the bundle.

Usage (from the repository root):
    python kaggle/build_bundle.py --out kaggle_stage --owner <your-kaggle-username>

It also writes dataset-metadata.json and the two kernel folders (smoke / full) that
`kaggle datasets create` and `kaggle kernels push` expect.
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_SLUG = "hrscreen-bundle"
BUNDLE_NAME = "hrscreen-bundle.dat"
MODEL_FOLDER = "all-MiniLM-L6-v2"


def find_model_snapshot() -> Path:
    """The locally cached Hugging Face snapshot of all-MiniLM-L6-v2 (downloaded by the first embedding run)."""
    hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots"
    snapshots = sorted(hub.glob("*"), key=lambda p: p.stat().st_mtime) if hub.exists() else []
    if not snapshots:
        raise SystemExit("Embedding model not cached locally. Run one embeddings benchmark first, or pass --model-dir.")
    return snapshots[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="staging folder (keep it outside the repo)")
    ap.add_argument("--owner", required=True, help="your Kaggle username")
    ap.add_argument("--model-dir", type=Path, help="local folder of the sentence-transformers model")
    ap.add_argument("--dataset-csv", type=Path, default=ROOT / "data" / "raw" / "resume-atlas" / "train.csv")
    ap.add_argument("--wheels", type=Path, help="optional folder of .whl files to install offline in the kernel")
    args = ap.parse_args()

    model_dir = args.model_dir or find_model_snapshot()
    out = Path(args.out)
    (out / "dataset").mkdir(parents=True, exist_ok=True)
    bundle = out / "dataset" / BUNDLE_NAME

    n = 0
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in (ROOT / "src" / "hrscreen").rglob("*"):
            if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc":
                zf.write(f, f.relative_to(ROOT).as_posix()); n += 1
        zf.write(args.dataset_csv, "data/raw/resume-atlas/train.csv"); n += 1
        for f in model_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f"models/{MODEL_FOLDER}/{f.relative_to(model_dir).as_posix()}"); n += 1
        for f in (args.wheels.glob("*.whl") if args.wheels else []):
            zf.write(f, f"wheels/{f.name}"); n += 1
    print(f"{n} files -> {bundle} ({bundle.stat().st_size / 1e6:.0f} MB)")

    (out / "dataset" / "dataset-metadata.json").write_text(json.dumps(
        {"id": f"{args.owner}/{DATASET_SLUG}", "title": DATASET_SLUG, "licenses": [{"name": "other"}]}, indent=2), encoding="utf-8")

    for mode in ("smoke", "full"):
        folder = out / f"kernel_{mode}"
        folder.mkdir(exist_ok=True)
        src = (ROOT / "kaggle" / "kernel.py").read_text(encoding="utf-8")
        (folder / "kernel.py").write_text(src.replace("MODE = None", f'MODE = "{mode}"'), encoding="utf-8")
        (folder / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{args.owner}/hrscreen-{mode}", "title": f"hrscreen-{mode}", "code_file": "kernel.py",
            "language": "python", "kernel_type": "script", "is_private": "true",
            "enable_gpu": "true", "enable_internet": "false",
            "dataset_sources": [f"{args.owner}/{DATASET_SLUG}"]}, indent=2), encoding="utf-8")
    print("wrote dataset-metadata.json, kernel_smoke/, kernel_full/ under", out)


if __name__ == "__main__":
    main()
