import pytest
import numpy as np
from am_combiner.combiners.common import get_unique_ids_and_blocking_fields
from am_combiner.utils.adjacency import (
    get_article_feature_adjacency_matrix,
    get_article_multi_feature_adjacency,
    get_feature_negative_edge_matrix,
    get_multi_feature_negative_edges,
)
from am_combiner.features.article import Features, Article
from am_combiner.features.sanction import Sanction, SanctionFeatures
from numpy.testing import assert_equal
from scipy.sparse import coo_matrix


class TestFeatureAdjacencyMatrixBuilding:
    def test_returns_empty_matrix_for_no_overlap(self):
        a1 = Article(entity_name="entity", article_text="text1", url="http1")
        a1.extracted_entities[Features.PERSON] = ["1", "2"]
        a2 = Article(entity_name="entity", article_text="text2", url="http2")
        a2.extracted_entities[Features.PERSON] = ["3", "4"]
        adj = get_article_feature_adjacency_matrix([a1, a2], Features.PERSON)
        assert_equal(adj.toarray(), [[0, 0], [0, 0]])

    def test_intersection_on_lower_case(self):
        a1 = Article(entity_name="entity", article_text="text1", url="http1")
        a1.extracted_entities[Features.PERSON] = ["john", "peter", "flow"]
        a2 = Article(entity_name="entity", article_text="text2", url="http2")
        a2.extracted_entities[Features.PERSON] = ["peter", "flow", "kl"]
        a3 = Article(entity_name="entity", article_text="text2", url="http2")
        a3.extracted_entities[Features.PERSON] = ["flow"]
        adj = get_article_feature_adjacency_matrix([a1, a2, a3], Features.PERSON)
        assert_equal(adj.toarray(), [[0, 2, 1], [2, 0, 1], [1, 1, 0]])

    def test_returns_empty_on_non_matching_case(self):
        a1 = Article(entity_name="entity", article_text="text1", url="http1")
        a1.extracted_entities[Features.PERSON] = ["peter", "john"]
        a2 = Article(entity_name="entity", article_text="text2", url="http2")
        a2.extracted_entities[Features.PERSON] = ["Peter", "John"]
        adj = get_article_feature_adjacency_matrix([a1, a2], Features.PERSON)
        assert_equal(adj.toarray(), [[0, 0], [0, 0]])

    def test_gracefully_recovers_from_non_matching_features(self):
        a1 = Article(entity_name="entity", article_text="text1", url="http1")
        a1.extracted_entities[Features.PERSON] = ["john", "peter", "flow"]
        a2 = Article(entity_name="entity", article_text="text2", url="http2")
        a2.extracted_entities[Features.ORG_CLEAN] = ["peter", "flow", "kl"]
        adj = get_article_feature_adjacency_matrix([a1, a2], Features.PERSON)
        assert_equal(adj.toarray(), [[0, 0], [0, 0]])


class TestMultiFeatureAdjacencyMatrixBuilding:
    @pytest.mark.parametrize(
        ("as_list", "expected"),
        ((False, [[0, 0], [0, 0]]), (True, [[[0, 0], [0, 0]], [[0, 0], [0, 0]]])),
    )
    def test_returns_empty_matrix_for_no_overlap(self, as_list, expected):
        a1 = Article(entity_name="entity", article_text="text1", url="http1")
        a1.extracted_entities[Features.PERSON] = ["1", "2"]
        a1.extracted_entities[Features.ORG_CLEAN] = ["11", "22"]
        a2 = Article(entity_name="entity", article_text="text2", url="http2")
        a2.extracted_entities[Features.PERSON] = ["3", "4"]
        a1.extracted_entities[Features.ORG_CLEAN] = ["33", "44"]
        adj = get_article_multi_feature_adjacency(
            [a1, a2], [Features.PERSON, Features.ORG_CLEAN], as_list=as_list
        )
        if as_list:
            adj = np.array([a.toarray() for a in adj])
        else:
            assert isinstance(adj, coo_matrix)
            adj = adj.toarray()
        assert_equal(adj, expected)

    @pytest.mark.parametrize(
        ("as_list", "expected"),
        (
            (False, [[0, 2, 3], [2, 0, 1], [3, 1, 0]]),
            (True, [[[0, 1, 1], [1, 0, 1], [1, 1, 0]], [[0, 1, 2], [1, 0, 0], [2, 0, 0]]]),
        ),
    )
    def test_returns_empty_matrix_for_overlap(self, as_list, expected):
        a1 = Article(entity_name="entity", article_text="text1", url="http1")
        a1.extracted_entities[Features.PERSON] = ["john", "peter", "flow"]
        a1.extracted_entities[Features.ORG] = ["apple", "netflix", "amazon"]
        a2 = Article(entity_name="entity", article_text="text2", url="http2")
        a2.extracted_entities[Features.PERSON] = ["apple", "flow", "kl"]
        a2.extracted_entities[Features.ORG] = ["apple"]
        a3 = Article(entity_name="entity", article_text="text2", url="http2")
        a3.extracted_entities[Features.PERSON] = ["flow"]
        a3.extracted_entities[Features.ORG] = ["netflix", "amazon"]
        adj = get_article_multi_feature_adjacency(
            [a1, a2, a3], [Features.PERSON, Features.ORG], as_list=as_list
        )
        if as_list:
            adj = np.array([a.toarray() for a in adj])
        else:
            adj = adj.toarray()
        assert_equal(adj, expected)


@pytest.mark.parametrize(
    ("input_objects", "expected_unique_ids", "expected_blocking_field"),
    (
        (
            [Article("John Smith", "", "www.bbc.co.uk/JohnSmith")],
            ["www.bbc.co.uk/JohnSmith"],
            ["John Smith"],
        ),
        (
            [Sanction("SM:IDH12", {}, "vessel"), Sanction("SM:IDH0", {}, "person")],
            ["SM:IDH12", "SM:IDH0"],
            ["vessel", "person"],
        ),
    ),
)
def test_get_unique_ids_and_blocking_fields(
    input_objects, expected_unique_ids, expected_blocking_field
):
    unique_ids, blocking_field = get_unique_ids_and_blocking_fields(input_objects)
    assert unique_ids == expected_unique_ids
    assert blocking_field == expected_blocking_field


@pytest.mark.parametrize(
    ("feature_values", "distance", "expected_negative_edge_sum"),
    (
        ([set([1992]), set([1993, 1994]), set([1995])], 1, 2),
        ([set([1992, 1993, 1994, 1995, 1996]), set([1992])], 0, 0),
        ([set([1992, 1993, 1994, 1995, 1996]), set([1994])], 0, 0),
        ([set([1992, 1993, 1994, 1995, 1996]), set([1996])], 0, 0),
        ([set([1992, 1993, 1994, 1995, 1996]), set([1999])], 0, 2),
        ([set([1992, 1993, 1994]), set([1995, 1996])], 0, 2),
        ([set(["FR", "GR"]), set(["FR"])], None, 0),
        ([set(["FR", "GR"]), set(["FR"]), set(["GR"])], None, 2),
        ([set(["FR", "GR"]), set(["FR"]), set()], None, 0),
    ),
)
def test_get_article_feature_negative_edge_matrix(
    feature_values, distance, expected_negative_edge_sum
):
    sanctions = []
    for feature_value in feature_values:
        sanction = Sanction("", {}, "")
        sanction.extracted_entities[SanctionFeatures.YOB] = feature_value
        sanctions.append(sanction)

    neg_adj = get_feature_negative_edge_matrix(sanctions, SanctionFeatures.YOB, distance)
    assert neg_adj.sum() == expected_negative_edge_sum


@pytest.mark.parametrize(
    ("feature1_values", "feature2_values", "distance", "expected_negative_edge_sum"),
    (
        ([set([1992]), set([1993, 1994])], [set(["FR"]), set(["GR"])], [2, None], 2),
        ([set(), set()], [set(), set()], [None, None], 0),
    ),
)
def test_get_article_multi_feature_negative_edges(
    feature1_values, feature2_values, distance, expected_negative_edge_sum
):
    sanctions = []
    for i, feature1_value in enumerate(feature1_values):
        sanction = Sanction("", {}, "")
        sanction.extracted_entities[SanctionFeatures.YOB] = feature1_value
        sanction.extracted_entities[SanctionFeatures.PRIMARY] = feature2_values[i]
        sanctions.append(sanction)

    negators = [(SanctionFeatures.YOB, distance[0]), (SanctionFeatures.PRIMARY, distance[1])]

    neg_adj = get_multi_feature_negative_edges(sanctions, negators)
    assert neg_adj.sum() == expected_negative_edge_sum
