from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from earthbridge.api.schemas import (
    DescriptorRetrieveRequest,
    HealthResponse,
    ModelInfoResponse,
    RetrievalResponse,
)
from earthbridge.api.service import RetrievalService

app = FastAPI(title="EarthBridge Retrieval API", version="0.1.0")
service = RetrievalService.from_environment()
STATIC_DIR = Path(__file__).resolve().parent / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", index_loaded=service.index_loaded)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(**service.model_info())


@app.post("/retrieve-descriptor", response_model=RetrievalResponse)
def retrieve_descriptor(request: DescriptorRetrieveRequest) -> RetrievalResponse:
    try:
        return RetrievalResponse(
            **service.retrieve_descriptor(
                descriptor=request.descriptor,
                top_k=request.top_k,
                exclude_ids=request.exclude_ids,
            )
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_image(
    image: Annotated[UploadFile, File(...)],
    query_modality: Annotated[str, Form(...)],
    target_modality: Annotated[str, Form(...)],
    top_k: Annotated[int, Form()] = 10,
) -> RetrievalResponse:
    try:
        content = await image.read()
        return RetrievalResponse(
            **service.retrieve_image(
                filename=image.filename or "query.tif",
                content=content,
                query_modality=query_modality,
                target_modality=target_modality,
                top_k=top_k,
            )
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/gallery/{sample_id}", include_in_schema=False)
def gallery_file(sample_id: str) -> FileResponse:
    path = service.gallery_path(sample_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Gallery sample not found")
    return FileResponse(path)
