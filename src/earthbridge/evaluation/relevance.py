from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from earthbridge.data.manifest import parse_label_string


class RelevanceMode(str, Enum):
    GEOGRAPHIC = "geographic"
    SEMANTIC = "semantic"
    PREDEFINED = "predefined"


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    modality: str
    pair_id: str = ""
    scene_id: str = ""
    geographic_group: str = ""
    labels: frozenset[str] = frozenset()

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> SampleRecord:
        return cls(
            sample_id=row.get("sample_id", ""),
            modality=row.get("modality", ""),
            pair_id=row.get("pair_id", ""),
            scene_id=row.get("scene_id", ""),
            geographic_group=row.get("geographic_group", ""),
            labels=parse_label_string(row.get("labels", "")),
        )


def jaccard_similarity(left: set[str] | frozenset[str], right: set[str] | frozenset[str]) -> float:
    if not left and not right:
        return 0.0

    union = left | right
    if not union:
        return 0.0

    return len(left & right) / len(union)


def is_geographically_relevant(query: SampleRecord, gallery: SampleRecord) -> bool:
    for field in ("pair_id", "scene_id", "geographic_group"):
        query_value = getattr(query, field)
        gallery_value = getattr(gallery, field)
        if query_value and gallery_value and query_value == gallery_value:
            return True
    return False


def is_semantically_relevant(
    query: SampleRecord,
    gallery: SampleRecord,
    threshold: float = 0.5,
) -> bool:
    return jaccard_similarity(query.labels, gallery.labels) >= threshold


def is_relevant(
    query: SampleRecord,
    gallery: SampleRecord,
    mode: RelevanceMode,
    predefined: Mapping[str, set[str]] | None = None,
    semantic_threshold: float = 0.5,
) -> bool:
    if mode == RelevanceMode.PREDEFINED:
        if predefined is None:
            raise ValueError("predefined relevance mode requires a relevance mapping")
        return gallery.sample_id in predefined.get(query.sample_id, set())

    if mode == RelevanceMode.GEOGRAPHIC:
        return is_geographically_relevant(query, gallery)

    if mode == RelevanceMode.SEMANTIC:
        return is_semantically_relevant(query, gallery, threshold=semantic_threshold)

    raise ValueError(f"Unsupported relevance mode: {mode}")


def relevant_ids(
    query: SampleRecord,
    gallery: Sequence[SampleRecord],
    mode: RelevanceMode,
    predefined: Mapping[str, set[str]] | None = None,
    semantic_threshold: float = 0.5,
    exclude_self: bool = True,
) -> set[str]:
    relevant: set[str] = set()
    for candidate in gallery:
        if exclude_self and candidate.sample_id == query.sample_id:
            continue
        if is_relevant(query, candidate, mode, predefined, semantic_threshold):
            relevant.add(candidate.sample_id)
    return relevant

