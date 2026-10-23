"""Kaggle kernel: run the hrscreen benchmark and fairness audits on a GPU, fully offline.

build_bundle.py replaces `MODE = None` below with "smoke" (small, quick end-to-end check of the
GPU, the library versions and every step) or "full" (the real run). Always run smoke first.
Any failing step raises, so Kaggle reports ERROR instead of COMPLETE.
"""
import glob
import os
import shutil
import subprocess
import sys
import zipfile

MODE = None

matches = glob.glob("/kaggle/input/**/hrscreen-bundle.dat", recursive=True)
if not matches:
    raise FileNotFoundError("hrscreen-bundle.dat not found under /kaggle/input/**")
print("bundle:", matches[0], "| mode:", MODE, flush=True)

WORK = "/kaggle/working/hrscreen"
os.makedirs(WORK, exist_ok=True)
with zipfile.ZipFile(matches[0]) as zf:
    zf.extractall(WORK)
os.chdir(WORK)
os.environ["PYTHONPATH"] = f"{WORK}/src"          # use the bundled code without pip-installing it
os.environ["HF_HUB_OFFLINE"] = "1"                 # the model is a local folder; never try the network
os.environ["TRANSFORMERS_OFFLINE"] = "1"
RESULTS = "/kaggle/working/results"
MODEL = f"{WORK}/models/all-MiniLM-L6-v2"


def step(name, cmd):
    log = f"/kaggle/working/{name}_log.txt"
    # braces group the whole command so tee captures all of it and pipefail reports its failure
    rc = subprocess.run(f"set -o pipefail; {{ {cmd}; }} 2>&1 | tee {log}", shell=True, executable="/bin/bash").returncode
    print(f"[{name}] exit {rc}", flush=True)
    if rc:
        raise RuntimeError(f"step '{name}' failed (exit {rc}); see {log}")


# ---- environment diagnostics (this is what the smoke run is for)
step("diagnose", "nvidia-smi || echo 'NO GPU VISIBLE'; python -c \""
     "import torch, platform; print('python', platform.python_version(), '| torch', torch.__version__, "
     "'| cuda available', torch.cuda.is_available()); "
     "print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)\"; "
     "pip list 2>/dev/null | grep -iE 'sentence|transformers|xgboost|scikit-learn|pandas|numpy|scipy|statsmodels|torch'")

wheels = glob.glob(f"{WORK}/wheels/*.whl")
if wheels:
    step("pip", f"pip install --no-index --no-deps {' '.join(wheels)}")
try:
    import sentence_transformers  # noqa: F401
except ImportError as exc:
    raise RuntimeError("sentence-transformers is missing in this image; add its wheel via build_bundle.py --wheels") from exc

hr = "python -m hrscreen"
emb = f"--embedding-model {MODEL} --device auto --embedding-cache /kaggle/working/emb.npz"
if MODE == "smoke":
    common = "--data resume-atlas --sample 600 --folds 3"
    models = "rules,tfidf,xgboost,embeddings,xgboost+emb"
    step("benchmark", f"{hr} benchmark {common} --models {models} {emb} --out {RESULTS}/benchmark")
    step("fairness_embeddings", f"{hr} fairness {common} --model embeddings {emb} --out {RESULTS}/fairness_embeddings")
elif MODE == "full":
    from concurrent.futures import ThreadPoolExecutor

    models = "rules,tfidf,xgboost,embeddings,xgboost+emb"

    def emb_for(stream):  # one embedding cache per stream so the two streams never write the same file
        return f"--embedding-model {MODEL} --device auto --embedding-cache /kaggle/working/emb_{stream}.npz"

    def stream_benchmark():
        step("benchmark", f"{hr} benchmark --data resume-atlas --folds 5 --models {models} {emb_for('bench')} --out {RESULTS}/benchmark")
        # GPU timing flatters the embedding models, so also time them on the CPU (small sample, timing only).
        step("cpu_timing", f"{hr} benchmark --data resume-atlas --sample 600 --folds 3 --models tfidf,embeddings "
                           f"--embedding-model {MODEL} --device cpu --out {RESULTS}/benchmark_cpu_timing")

    def stream_fairness():
        for model in ("tfidf", "embeddings", "xgboost+emb"):
            tag = model.replace("+", "_")
            step(f"fairness_{tag}", f"{hr} fairness --data resume-atlas --folds 5 --model {model} {emb_for('fair')} --out {RESULTS}/fairness_{tag}")

    # Kaggle gives 4 cores and the two streams are independent, so run them side by side.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(stream_benchmark), pool.submit(stream_fairness)]
        for fut in futures:
            fut.result()   # re-raises the first failure so the kernel ends in ERROR, not COMPLETE
else:
    raise RuntimeError("MODE must be 'smoke' or 'full' (build_bundle.py sets it)")

expected = ["benchmark/benchmark.md", "benchmark/benchmark_summary.csv", "benchmark/benchmark_meta.json"]
missing = [f for f in expected if not os.path.exists(f"{RESULTS}/{f}")]
if missing:
    raise RuntimeError(f"missing outputs: {missing}")

with zipfile.ZipFile("/kaggle/working/hrscreen_results.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(RESULTS):
        for name in files:
            path = os.path.join(root, name)
            zf.write(path, os.path.relpath(path, RESULTS))
os.chdir("/kaggle/working")
shutil.rmtree(WORK)          # keep the output listing to the results zip + logs
print("done", flush=True)
