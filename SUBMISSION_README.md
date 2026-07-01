# EarthBridge Final Hackathon Submission

This folder is the offline-ready EarthBridge submission package for cross-modal satellite image retrieval on BEN-14K.

## Verified Artifact Set

The final artifact bundle contains:

```text
artifacts/checkpoints/baseline_pair.pt
artifacts/descriptors/gallery.npy
artifacts/descriptors/gallery_ids.json
artifacts/indexes/gallery.index
artifacts/indexes/gallery_ids.json
artifacts/manifests/test.csv
artifacts/reports/evaluation_summary.json
artifacts/reports/direction_metrics.csv
artifacts/reports/geographic_evaluation_summary.json
artifacts/reports/geographic_direction_metrics.csv
artifacts/reports/final_submission_metrics.json
artifacts/reports/latency_summary.json
```

Verification summary:

- Checkpoint loads successfully and contains `model_state_dict`.
- Descriptor matrix is finite with shape `6496 x 128`.
- FAISS index is trained, dimension `128`, and contains `6496` gallery vectors.
- Descriptor IDs and FAISS index IDs match exactly.
- Test manifest contains `6496` images, `3248` paired Sentinel-1/Sentinel-2 samples, labels for every row, and the `test` split only.

## Final Metrics

Use `artifacts/reports/final_submission_metrics.json` as the authoritative judging summary.

| Metric | Value |
| --- | ---: |
| Cross-modal F1@5 | 19.53% |
| Cross-modal F1@10 | 11.59% |
| Full-validation Recall@1 | 53.03% |
| Full-validation Recall@10 | 70.08% |
| Mean retrieval latency | about 0.146 ms |

The geographic cross-modal metrics are also present in `artifacts/reports/geographic_evaluation_summary.json` and `artifacts/reports/geographic_direction_metrics.csv`. The separate `latency_summary.json` file is a standalone 100-query benchmark; the final submission metric uses the evaluation-time mean in `final_submission_metrics.json`.

## Architecture

EarthBridge builds a sensor-independent embedding space for Sentinel-1 SAR and Sentinel-2 multispectral patches.

1. BEN-14K metadata pairs Sentinel-2 `patch_id` with the associated Sentinel-1 `s1_name`.
2. Multi-band TIFF loading uses `rasterio`, preserving 2-band SAR and 10-band multispectral inputs.
3. Each modality is normalized band-wise before encoding.
4. Modality-specific input adapters feed a compact GroupNorm convolutional encoder and projection head.
5. Training uses exact-pair contrastive retrieval so matching SAR/Sentinel-2 patches are pulled together.
6. The final gallery descriptors are precomputed and searched with FAISS for low-latency top-k retrieval.
7. A FastAPI demo serves upload-based retrieval and gallery previews from the exported artifacts.

## Run Instructions

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate earthbridge
pip install -r requirements.txt
pip install -r requirements-ml.txt
```

Verify the packaged artifacts:

```bash
python scripts/verify_artifacts.py --artifact-root artifacts
```

Start the local demo:

```bash
uvicorn earthbridge.api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/
```

Demo flow:

1. Upload or select a Sentinel-2 multispectral query and retrieve Sentinel-1 SAR matches.
2. Upload or select a Sentinel-1 SAR query and retrieve Sentinel-2 multispectral matches.
3. Show `artifacts/reports/final_submission_metrics.json`.
4. Show the sub-millisecond retrieval latency and explain that FAISS searches precomputed descriptors.

## Upload Checklist

Upload the clean final ZIP produced from this package, not the Git repository alone. It must include:

- Source code: `src/`, `scripts/`, `configs/`, `notebooks/`, `tests/`.
- Project files: `README.md`, `SUBMISSION_README.md`, `pyproject.toml`, `environment.yml`, `requirements.txt`, `requirements-ml.txt`.
- Documentation: `docs/`.
- Final artifacts: `artifacts/checkpoints/`, `artifacts/descriptors/`, `artifacts/indexes/`, `artifacts/manifests/`, `artifacts/reports/`.
- No `.git/`, cache folders, raw BEN-14K dataset, or temporary verification folders.
