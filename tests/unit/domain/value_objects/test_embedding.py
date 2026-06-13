import pytest

from app.domain.value_objects.embedding import Embedding


def test_embedding_rejects_empty_vector():
    with pytest.raises(ValueError, match="must not be empty"):
        Embedding(vector=(), model_id="bge-small-en-v1.5")


@pytest.mark.parametrize(
    "non_finite", [float("inf"), float("-inf"), float("nan")]
)
def test_embedding_rejects_non_finite_components(non_finite):
    with pytest.raises(ValueError, match="must all be finite"):
        Embedding(vector=(0.1, non_finite, 0.3), model_id="bge-small-en-v1.5")


def test_embedding_rejects_empty_model_id():
    with pytest.raises(ValueError, match="model_id"):
        Embedding(vector=(0.1, 0.2), model_id="")


def test_embedding_dimensions_matches_vector_length():
    embedding = Embedding(vector=(0.1, 0.2, 0.3), model_id="bge-small-en-v1.5")

    assert embedding.dimensions == 3
