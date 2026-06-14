"""Port: text -> vector. FREE (local) and PAID implementations (ADR-004)."""

from abc import ABC, abstractmethod

from app.domain.value_objects.embedding import Embedding


class IEmbeddingService(ABC):
    """Turns text into dense vectors tied to a specific model."""

    @abstractmethod
    async def embed(self, text: str) -> Embedding:
        """Vectorize a single text.

        The returned Embedding carries `model_id` so consumers can verify
        it belongs to the expected vector space (ADR-004).

        Raises:
            EmbeddingError: the provider failed (API error, local model
                unavailable, input over the model's token limit).
        """
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[Embedding]:
        """Vectorize many texts at once.

        Guarantee: output is in the same order as input and of the same
        length; `result[i]` is the embedding of `texts[i]`.

        Raises:
            EmbeddingError: the provider failed for the batch.
        """
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Dimension of the vectors this service produces.

        The repository needs it to create the collection with the right
        size and to validate at startup that an existing collection
        matches (ADR-004).
        """
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Stable identifier of the model (e.g. "bge-small-en-v1.5").

        Used to compose the Qdrant collection name (`signals__<model_id>`)
        so vectors from different models never share a collection.
        """
        ...
