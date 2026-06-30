from __future__ import annotations

from fastapi import FastAPI, HTTPException

from earthbridge.api.schemas import (
    DescriptorRetrieveRequest,
    HealthResponse,
    ModelInfoResponse,
    RetrievalResponse,
)
from earthbridge.api.service import RetrievalService

app = FastAPI(title="EarthBridge Retrieval API", version="0.1.0")
service = RetrievalService.from_environment()


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

