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

Both notebooks first run the real-data exact-overfit gate:

```bash
python scripts/run_tiny_overfit_matrix.py \
  --manifest data/manifests/train.csv \
  --root-dir /kaggle/input/datasets/narendraaironi/bigearthnet-14k \
  --left-modality multispectral \
  --right-modality sar \
  --pair-count 128 \
  --batch-size 128 \
  --epochs 100 \
  --device cuda
```

Do not start full training unless this writes:

```text
artifacts/tiny_overfit/best_tiny_overfit_config.json
```

After that gate passes, the direct full-pipeline command is:

```bash
python scripts/run_cloud_pipeline.py \
  --data-raw /kaggle/input/datasets/narendraaironi/bigearthnet-14k/BEN_14k \
  --left-modality multispectral \
  --right-modality sar \
  --batch-size 128 \
  --projection-dropout 0 \
  --semantic-loss-weight 0 \
  --hard-negative-loss-weight 0 \
  --diagnostic-sample-count 128 \
  --seed 42 \
  --device cuda \
  --export-zip artifacts/earthbridge_export.zip
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
artifacts/manifests/test.csv
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

If the notebook stops before training, inspect:

```text
artifacts/reports/manifest_samples_check.json
artifacts/reports/manifest_train_check.json
artifacts/reports/manifest_test_check.json
```

These reports catch missing images, missing labels, duplicate IDs, and zero paired samples.

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
