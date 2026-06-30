# EarthBridge

EarthBridge is a hackathon-oriented cross-modal satellite image retrieval system for same-modal and cross-modal search across optical, SAR, and multispectral imagery.

The project is built evaluator-first because the scoring depends on:

- same-modal `F1@5`
- same-modal `F1@10`
- cross-modal `F1@5`
- cross-modal `F1@10`
- average retrieval time per query

## Build Order

1. Verify metrics, relevance logic, and leakage-free splits.
2. Build a canonical manifest for every image.
3. Train a simple paired contrastive retrieval baseline.
4. Precompute gallery descriptors and search with FAISS or the NumPy fallback.
5. Add dual heads, semantic positives, and hard negatives only after the baseline works.
6. Add FastAPI and a browser demo after `/retrieve` is correct.

## First MVP

The first working version should support:

- optical to SAR retrieval
- SAR to optical retrieval
- optical to optical retrieval
- SAR to SAR retrieval
- top-5 and top-10 outputs
- F1@5, F1@10, and latency reporting

## Repository Layout

```text
configs/              configuration files
data/manifests/       canonical CSV manifests and split files
data/relevance/       optional predefined relevance maps
scripts/              runnable project scripts
src/earthbridge/      Python package
tests/                correctness tests
artifacts/            generated reports, descriptors, indexes, checkpoints
```

## Setup

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate earthbridge
```

If the environment already exists, install or refresh the lightweight development dependencies:

```bash
pip install -r requirements.txt
```

For model training, install PyTorch using the official selector for your GPU/CUDA setup, then add FAISS. Keep the project on Python 3.10 or 3.11 for PyTorch and FAISS compatibility.

On this laptop, Windows reports Intel Iris Xe graphics and no visible NVIDIA CUDA GPU, so the local setup uses CPU PyTorch:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-ml.txt
```

For faster training, use the same repository on Colab, Kaggle, or a cloud GPU with the matching CUDA PyTorch install command.

## Run Tests

```bash
pytest
```

Do not start model training until the metric, relevance, and split tests pass.

## First Data Commands

After placing the dataset under `data/raw/`, inspect it:

```bash
python scripts/inspect_data.py --input data/raw --output-dir artifacts/reports
```

Then create the first canonical manifest:

```bash
python scripts/build_manifest.py --input data/raw --output data/manifests/samples.csv
```

Generate baseline descriptors from a manifest:

```bash
python scripts/generate_descriptors.py \
  --manifest data/manifests/samples.csv \
  --root-dir . \
  --output-dir artifacts/descriptors \
  --name gallery
```

Build a FAISS index from those descriptors:

```bash
python scripts/build_indexes.py \
  --descriptors artifacts/descriptors/gallery.npy \
  --ids artifacts/descriptors/gallery_ids.json \
  --output-index artifacts/indexes/gallery.index
```

Train the first paired baseline after splits are available:

```bash
python scripts/train_baseline.py \
  --manifest data/manifests/train.csv \
  --root-dir . \
  --left-modality optical_rgb \
  --right-modality sar \
  --epochs 5 \
  --output-checkpoint artifacts/checkpoints/baseline_pair.pt
```

Evaluate descriptor retrieval:

```bash
python scripts/evaluate_descriptors.py \
  --manifest data/manifests/test.csv \
  --descriptors artifacts/descriptors/gallery.npy \
  --ids artifacts/descriptors/gallery_ids.json \
  --relevance-mode semantic
```

Benchmark search latency:

```bash
python scripts/benchmark_latency.py \
  --descriptors artifacts/descriptors/gallery.npy \
  --ids artifacts/descriptors/gallery_ids.json \
  --top-k 10
```

## Demo Index And API Smoke Test

Create a self-contained synthetic smoke-demo bundle:

```bash
python scripts/create_demo_index.py --output-dir artifacts/demo
```

Point the API at those smoke-demo artifacts in PowerShell:

```powershell
$env:EARTHBRIDGE_CHECKPOINT_PATH='artifacts/demo/checkpoints/baseline_pair.pt'
$env:EARTHBRIDGE_INDEX_PATH='artifacts/demo/indexes/gallery.index'
$env:EARTHBRIDGE_IDS_PATH='artifacts/demo/indexes/gallery_ids.json'
$env:EARTHBRIDGE_GALLERY_MANIFEST='artifacts/demo/manifests/test.csv'
$env:EARTHBRIDGE_GALLERY_ROOT='artifacts/demo'
```

Start the API:

```bash
uvicorn earthbridge.api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/
```

The smoke bundle is only for proving the local API, upload UI, checkpoint loading, TIFF previews, and FAISS retrieval path. Use trained Kaggle or Colab artifacts for final metrics.

After importing trained artifacts, the local demo reads by default:

```text
artifacts/checkpoints/baseline_pair.pt
artifacts/indexes/gallery.index
artifacts/indexes/gallery_ids.json
artifacts/manifests/test.csv
```

Open the upload UI at:

```text
http://127.0.0.1:8000/
```

## Cloud Training

Use cloud notebooks for heavy training, then bring only verified artifacts back to this laptop:

```text
notebooks/kaggle_train_baseline.ipynb
notebooks/colab_train_baseline.ipynb
```

After downloading `artifacts/earthbridge_export.zip` from Kaggle or Colab, extract it into the project root and verify:

```bash
python scripts/verify_artifacts.py --artifact-root artifacts
```

See [docs/CLOUD_TRAINING.md](docs/CLOUD_TRAINING.md) for the full cloud-to-local workflow.
