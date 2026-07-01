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

## Verified Final Submission Artifacts

- `earthbridge_final_submission.zip` was inspected and extracted into an isolated verification folder.
- `scripts/verify_artifacts.py --artifact-root artifacts` passed on the extracted artifact tree.
- The checkpoint loads and contains `model_state_dict`.
- The descriptor matrix has shape `6496 x 128`, all finite values, and matches the FAISS index count.
- The FAISS index is trained, dimension `128`, and contains `6496` gallery vectors.
- Descriptor IDs and FAISS index IDs match exactly.
- The test manifest contains `6496` images, `3248` paired samples, both `sar` and `multispectral` modalities, and labels for every row.

Final metrics from `artifacts/reports/final_submission_metrics.json`:

- Cross-modal F1@5: 19.53%
- Cross-modal F1@10: 11.59%
- Full-validation Recall@1: 53.03%
- Full-validation Recall@10: 70.08%
- Mean retrieval latency: about 0.146 ms

Remaining manual action: upload the clean final submission ZIP to the hackathon portal.

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
