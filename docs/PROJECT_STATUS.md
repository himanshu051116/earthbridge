# EarthBridge Project Status

## Completed

- Dataset inspection handles normal images and multi-band TIFF/JP2 rasters.
- BEN-14K folders are recognized as `sar` and `multispectral`.
- BEN-14K metadata pairing uses Sentinel-2 `patch_id` and associated `s1_name`.
- Multi-label land-cover labels are carried into manifests when metadata is available.
- Existing `train`, `validation`, and `test` folders are preserved.
- Train, descriptor generation, FAISS indexing, evaluation, latency, artifact export, and artifact verification scripts are in place.
- Manifest validation catches missing files, missing labels, duplicate sample IDs, and zero paired samples before training.
- Kaggle and Colab notebooks are configured for cloud training and artifact export.
- `scripts/run_cloud_pipeline.py` runs the full cloud flow from raw data to export zip.
- FastAPI serves descriptor retrieval, image upload retrieval, model info, health checks, and gallery previews.
- The browser demo can upload query images and show ranked results with PNG previews.
- `scripts/create_demo_index.py` creates a local synthetic smoke-demo bundle for API/UI testing.

## Remaining Before Final Submission

- Run `notebooks/kaggle_train_baseline.ipynb` on the attached BEN-14K Kaggle dataset.
- Download `artifacts/earthbridge_export.zip` from Kaggle.
- Extract the zip into the local project root.
- Run `python scripts/verify_artifacts.py --artifact-root artifacts`.
- Run the API using the real trained artifacts, not the synthetic smoke bundle.
- Capture final `direction_metrics.csv`, `evaluation_summary.json`, and `latency_summary.json` for presentation.

## Main Risks

- The laptop is CPU-only, so training must stay on Kaggle or Colab.
- Browser previews need PNG conversion because TIFF/SAR/multispectral files are not reliably displayable directly.
- Smoke-demo artifacts prove plumbing only; they are not model-quality evidence.
- Final judging should not depend on internet access, Kaggle runtime availability, or live dataset downloads.

## Recommended Final Demo Flow

1. Show the problem: same-modal and cross-modal retrieval across Sentinel-1 SAR and Sentinel-2 multispectral data.
2. Show the manifest and pairing method: BEN metadata, not alphabetical matching.
3. Show final metrics from `artifacts/reports/direction_metrics.csv`.
4. Open `http://127.0.0.1:8000/`.
5. Upload a Sentinel-2 query and retrieve Sentinel-1 results.
6. Upload a Sentinel-1 query and retrieve Sentinel-2 results.
7. Show latency summary and explain that retrieval uses precomputed FAISS descriptors.
