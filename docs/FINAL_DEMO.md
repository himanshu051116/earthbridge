# Final Demo Runbook

This is the hackathon-safe demo path.

## 0. Local Smoke Check

Before waiting on Kaggle training, prove the API and UI path locally:

```powershell
python scripts/create_demo_index.py --output-dir artifacts/demo
$env:EARTHBRIDGE_CHECKPOINT_PATH='artifacts/demo/checkpoints/baseline_pair.pt'
$env:EARTHBRIDGE_INDEX_PATH='artifacts/demo/indexes/gallery.index'
$env:EARTHBRIDGE_IDS_PATH='artifacts/demo/indexes/gallery_ids.json'
$env:EARTHBRIDGE_GALLERY_MANIFEST='artifacts/demo/manifests/test.csv'
$env:EARTHBRIDGE_GALLERY_ROOT='artifacts/demo'
uvicorn earthbridge.api.main:app --reload
```

Open `http://127.0.0.1:8000/` and upload one of the files under `artifacts/demo/images/`.
These are synthetic smoke artifacts, not final metrics.

## 1. Train Online

Run:

```text
notebooks/kaggle_train_baseline.ipynb
```

The notebook is configured for:

```text
/kaggle/input/datasets/narendraaironi/bigearthnet-14k/BEN_14k
```

Download:

```text
artifacts/earthbridge_export.zip
```

## 2. Restore Artifacts Locally

Extract the zip into the project root. It should restore files under:

```text
artifacts/checkpoints/
artifacts/descriptors/
artifacts/indexes/
artifacts/manifests/
artifacts/reports/
```

## 3. Verify Artifacts

```bash
conda activate earthbridge
cd /d E:\Isro
python scripts/verify_artifacts.py --artifact-root artifacts
```

## 4. Run API And UI

```bash
uvicorn earthbridge.api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/
```

Useful API docs:

```text
http://127.0.0.1:8000/docs
```

## 5. Demo Checklist

- Upload a Sentinel-2 multispectral query and retrieve Sentinel-1 SAR results.
- Upload a Sentinel-1 SAR query and retrieve Sentinel-2 multispectral results.
- Show `artifacts/reports/direction_metrics.csv`.
- Show `artifacts/reports/latency_summary.json`.
- Keep the demo local. Do not rely on Kaggle during judging.
