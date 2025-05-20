import pytest
import numpy as np
from am_combiner.splitters.common import DeleteNegativeEdgesSplitter, ColourNegativeEdgesSplitter


@pytest.mark.parametrize(
    ("pos_adj", "neg_adj", "subgraph_num"),
    (
        (np.ones((7, 7)), np.ones((7, 7)), 7),
        (np.ones((7, 7)), np.zeros((7, 7)), 1),
        (
            np.ones((3, 3)),
            np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
            1,
        ),
        (
            np.ones((3, 3)),
            np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]]),
            2,
        ),
    ),
)
def test_break_subgraph_apart(pos_adj, neg_adj, subgraph_num):
    combiner = DeleteNegativeEdgesSplitter([], [])
    combiner.cluster_counter = 0
    n = pos_adj.shape[0]
    combiner.cluster_ids = [0] * n

    combiner.break_subgraph(pos_adj, neg_adj, list(range(n)))
    assert combiner.cluster_counter == subgraph_num


@pytest.mark.parametrize(
    ("pos_adj", "neg_adj", "expected_clustering"),
    (
        (np.ones((7, 7)), np.ones((7, 7)), list(range(7))),
        (np.ones((7, 7)), np.zeros((7, 7)), [0] * 7),
        (
            np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]]),
            np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
            [0, 1, 0],
        ),
        (
            np.ones((3, 3)),
            np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]]),
            [0, 1, 1],
        ),
    ),
)
def test_break_subgraph_apart_by_colouring(pos_adj, neg_adj, expected_clustering):
    combiner = ColourNegativeEdgesSplitter([], [])
    combiner.cluster_counter = 0
    n = pos_adj.shape[0]
    combiner.cluster_ids = [0] * n

    combiner.break_subgraph(pos_adj, neg_adj, list(range(n)))
    assert combiner.cluster_ids == expected_clustering
