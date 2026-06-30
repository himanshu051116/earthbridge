# Final Demo Runbook

This is the hackathon-safe demo path.

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

