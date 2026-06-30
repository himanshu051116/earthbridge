# Cloud Training Workflow

Use Kaggle or Colab for heavy training, then bring only final artifacts back to the laptop.

## Safe Hackathon Rule

Do not depend on cloud training during the final demo. Train online, export artifacts, verify them locally, then demo from local files.

## Recommended Flow

1. Upload or clone this repository into Kaggle or Colab.
2. Attach the dataset or a dataset subset.
3. Run one of:

```text
notebooks/kaggle_train_baseline.ipynb
notebooks/colab_train_baseline.ipynb
```

4. Download:

```text
artifacts/earthbridge_export.zip
```

5. Extract it into the local project root so it restores:

```text
artifacts/checkpoints/baseline_pair.pt
artifacts/descriptors/gallery.npy
artifacts/descriptors/gallery_ids.json
artifacts/indexes/gallery.index
artifacts/indexes/gallery_ids.json
artifacts/reports/evaluation_summary.json
artifacts/reports/direction_metrics.csv
artifacts/reports/latency_summary.json
```

6. Verify locally:

```bash
python scripts/verify_artifacts.py --artifact-root artifacts
```

7. Start the local API:

```bash
uvicorn earthbridge.api.main:app --reload
```

## Kaggle Notes

Prefer Kaggle when the dataset is large. Attach the dataset to the notebook and set:

```python
DATA_RAW = Path("/kaggle/input/YOUR_DATASET_FOLDER")
IMAGE_ROOT = DATA_RAW.parent
```

## Colab Notes

Prefer Colab only for smaller subsets or quick experiments. Put the dataset in Drive and set:

```python
DATA_RAW = Path("/content/drive/MyDrive/earthbridge/data/raw")
IMAGE_ROOT = DATA_RAW.parent
```

## What To Bring Back

The minimum final demo set is:

```text
checkpoint + descriptors + FAISS index + reports
```

That makes the final demo independent of internet, GPU availability, and cloud runtime limits.

