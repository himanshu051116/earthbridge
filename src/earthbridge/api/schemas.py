from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool


class ModelInfoResponse(BaseModel):
    name: str
    embedding_dim: int | None
    index_size: int
    index_type: str
    model_loaded: bool = False
    gallery_metadata_loaded: bool = False


class DescriptorRetrieveRequest(BaseModel):
    descriptor: list[float] = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    exclude_ids: list[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    rank: int
    gallery_id: str
    similarity: float
    modality: str = ""
    labels: str = ""
    image_path: str = ""
    thumbnail: str = ""


class RetrievalResponse(BaseModel):
    retrieval_type: str
    retrieval_time_ms: float
    results: list[RetrievalResult]
