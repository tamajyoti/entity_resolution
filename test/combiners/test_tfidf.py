import numpy as np
import pytest
from scipy import sparse

from am_combiner.combiners.tfidf import (
    TFIDFCosineSimilarityCombiner,
    TFIDFAndFeaturesCosineSimilarityCombiner,
)
from am_combiner.features.article import Features, Article


def assert_array(actual_similarities, expected_similarity):
    for i in range(len(actual_similarities)):
        for j in range(len(actual_similarities)):
            assert round(actual_similarities[i][j], 2) == expected_similarity[i][j]


@pytest.mark.parametrize(
    "tfidf_combiner, expected_similarity, expected_enhanced_similarity, expected_adjacency_matrix",
    [
        (
            TFIDFCosineSimilarityCombiner(0.2, Features.TFIDF_FULL_TEXT),
            [np.array([1.0, 0.45]), np.array([0.45, 1.0])],
            [np.array([1.0, 0.45]), np.array([0.45, 1.0])],
            [np.array([1.0, 1.0]), np.array([1.0, 1.0])],
        ),
        (
            TFIDFAndFeaturesCosineSimilarityCombiner(
                [Features.LOC], 0.3, 30, Features.TFIDF_FULL_TEXT
            ),
            [np.array([1.0, 0.45]), np.array([0.45, 1.0])],
            [np.array([1.0, 0.48]), np.array([0.48, 1.0])],
            [np.array([1.0, 1.0]), np.array([1.0, 1.0])],
        ),
    ],
)
def test_tfidf_combiners_similarity_and_adjacency(
    tfidf_combiner, expected_similarity, expected_enhanced_similarity, expected_adjacency_matrix
):
    first_article = Article("Some Name", "Some first text", "some.first.url")
    first_article.extracted_entities[Features.TFIDF_FULL_TEXT] = sparse.csr_matrix(
        np.array([1, 0, 2]).reshape(1, -1)
    )
    first_article.extracted_entities[Features.LOC] = ["Michigan ", "Hollywood ", "Toronto"]

    second_article = Article("Some Name", "Some second text", "some.second.url")
    second_article.extracted_entities[Features.TFIDF_FULL_TEXT] = sparse.csr_matrix(
        np.array([2, 0, 0]).reshape(1, -1)
    )
    second_article.extracted_entities[Features.LOC] = ["Romania ", "Toronto", "Italy"]

    input_entities = [first_article, second_article]

    actual_similarities = tfidf_combiner._get_pairwise_similarities(input_entities)
    assert_array(actual_similarities, expected_similarity)

    actual_enhanced_similarities = tfidf_combiner._enhance_pairwise_similarities(
        actual_similarities, input_entities
    )
    assert_array(actual_enhanced_similarities, expected_enhanced_similarity)

    actual_adjacency_matrix = tfidf_combiner._get_adjacency_from_similarities(
        actual_enhanced_similarities
    )
    assert_array(actual_adjacency_matrix, expected_adjacency_matrix)

    assert tfidf_combiner.mongo_client is None
