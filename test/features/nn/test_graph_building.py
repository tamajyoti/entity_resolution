import pytest
import numpy as np
from am_combiner.features.article import Article
from am_combiner.features.nn.common import (
    articles_to_homogeneous_graph,
    articles_to_hetero_graph,
    get_node_embeddings,
    UnknownNumericFeatureTypeException,
)

from am_combiner.utils.adjacency import get_article_multi_feature_adjacency
from numpy.testing import assert_array_equal
from scipy.sparse import coo_matrix


class TestGraphBuilding:
    def test_articles_to_node_embeddings_fails_on_non_existent_feature(self):
        a = Article(entity_name="A", article_text="A B C", url="http://lol.com")
        with pytest.raises(KeyError):
            get_node_embeddings([a], "THIS ONE IS NOT THERE")

    def test_articles_to_node_embeddings_raise_error_on_unknown_feature_type(self):
        a = Article(entity_name="A", article_text="A B C", url="http://lol.com")
        a.extracted_entities["A"] = [1, 2, 3, "WHAT IS IT"]
        with pytest.raises(UnknownNumericFeatureTypeException):
            get_node_embeddings([a], "A")

    def test_node_embeddings_have_correct_shape(self):
        a1 = Article(entity_name="A", article_text="A B C", url="http://lol.com")
        a1.extracted_entities["A"] = np.ones(shape=(100,))
        a2 = Article(entity_name="B", article_text="CCC", url="http://lol1.com")
        a2.extracted_entities["A"] = np.zeros(shape=(100,))
        embeddings = get_node_embeddings([a1, a2], "A")
        assert embeddings.num_embeddings == 2
        assert embeddings.embedding_dim == 100
        assert_array_equal(
            np.stack((a1.extracted_entities["A"], a2.extracted_entities["A"])),
            embeddings.weight.numpy(),
        )

    @pytest.mark.parametrize(
        [
            "expected_num_nodes",
            "expected_num_edges",
            "expected_adj_matrix",
            "extracted_entities",
            "features_to_use",
        ],
        [
            (
                2,
                1,
                np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
                [
                    {"A": np.ones(shape=(100,)), "B": {1, 2}},
                    {"A": np.zeros(shape=(100,)), "B": {3, 4}},
                ],
                ("B",),
            ),
            (
                2,
                2,
                np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
                [
                    {"A": np.ones(shape=(100,)), "B": {1, 2}},
                    {"A": np.zeros(shape=(100,)), "B": {2, 3}},
                ],
                ("B",),
            ),
            (
                3,
                4,
                np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32),
                [
                    {"A": np.ones(shape=(100,)), "B": {1, 2}, "C": set()},
                    {"A": np.zeros(shape=(100,)), "B": {2, 3}, "C": {10}},
                    {"A": np.zeros(shape=(100,)), "B": {22, 33}, "C": {10, 20}},
                ],
                ("B", "C"),
            ),
        ],
    )
    def test_homo_graph_has_correct_structure(
        self,
        expected_num_nodes,
        expected_num_edges,
        expected_adj_matrix,
        extracted_entities,
        features_to_use,
    ):
        articles = []
        for ef in extracted_entities:
            a = Article(entity_name="A", article_text="A B C", url="http://lol.com")
            a.extracted_entities = ef
            articles.append(a)
        G, _ = articles_to_homogeneous_graph(articles, "A", features_to_use)
        assert G.num_nodes() == expected_num_nodes
        assert G.num_edges() == expected_num_edges
        assert_array_equal(
            G.adjacency_matrix().to_dense().numpy(),
            expected_adj_matrix,
        )

    @pytest.mark.parametrize(
        [
            "expected_weights",
            "extracted_entities",
            "features_to_use",
        ],
        [
            (
                np.array([3.0, 4.0, 3.0, 5.0, 4.0, 5.0]),
                [
                    {"A": np.ones(shape=(10,)), "B": {1, 2, 3, 5}},
                    {"A": np.zeros(shape=(10,)), "B": {1, 2, 3, 4}},
                    {"A": np.zeros(shape=(10,)), "B": {1, 2, 3, 4, 5}},
                ],
                ("A", "B"),
            ),
        ],
    )
    def test_homo_graph_has_correct_weighted_edges(
        self,
        expected_weights,
        extracted_entities,
        features_to_use,
    ):
        articles = []
        for ef in extracted_entities:
            a = Article(entity_name="A", article_text="A B C", url="http://lol.com")
            a.extracted_entities = ef
            articles.append(a)
        G, _ = articles_to_homogeneous_graph(articles, "A", features_to_use)
        assert_array_equal(
            G.edata["weight"].detach().numpy(),
            expected_weights,
        )

    @pytest.mark.parametrize(
        [
            "extracted_entities",
            "features_to_use",
        ],
        [
            (
                [
                    {"A": np.ones(shape=(10,)), "B": {1, 2}, "C": {1, 2}},
                    {"A": np.zeros(shape=(10,)), "B": {1, 2, 3, 4}, "C": {3}},
                    {"A": np.zeros(shape=(10,)), "B": {1, 2, 3, 4, 5}, "C": {1, 2, 3}},
                ],
                ("B", "C"),
            ),
        ],
    )
    def test_hetero_graph_has_correct_weighted_edges(
        self,
        extracted_entities,
        features_to_use,
    ):
        articles = []
        n = len(extracted_entities)
        for ef in extracted_entities:
            a = Article(entity_name="A", article_text="A B C", url="http://lol.com")
            a.extracted_entities = ef
            articles.append(a)
        G, _ = articles_to_hetero_graph(articles, "A", features_to_use)

        # Reconstruct adjacency matrix from dgl Graph and compare it to adjacency matrix produced.
        # This will verify that weights are assigned to correct edges in the Graph
        for feature_to_use in features_to_use:
            adjacency_from_graph = coo_matrix(
                (G[feature_to_use].edata["weight"], G[feature_to_use].edges()),
                shape=(n, n),
                dtype=np.int32,
            )
            adjacency_matrix = get_article_multi_feature_adjacency(articles, [feature_to_use])
            assert_array_equal(
                adjacency_matrix.toarray(),
                adjacency_from_graph.toarray(),
            )
