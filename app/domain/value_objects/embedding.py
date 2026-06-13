"""Vector representation of a text, produced by an IEmbeddingService."""

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Embedding:
    """An L2-normalizable dense vector tied to the model that produced it.

    Invariants: vector is non-empty and every component is a finite number.
    `model_id` identifies the embedding model (e.g. "bge-small-en-v1.5");
    vectors from different models live in incompatible spaces and must not
    be compared (see ADR-004).
    """

    vector: tuple[float, ...]
    model_id: str

    def __post_init__(self) -> None:
        if not self.vector:
            raise ValueError("Embedding.vector must not be empty")
        if not all(math.isfinite(component) for component in self.vector):
            raise ValueError("Embedding.vector components must all be finite")
        if not self.model_id:
            raise ValueError("Embedding.model_id must not be empty")

    @property
    def dimensions(self) -> int:
        return len(self.vector)
