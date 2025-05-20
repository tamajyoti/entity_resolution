import numpy as np
import pytest

from am_combiner.combiners.incremental_clustering import (
    EntityCluster,
    PairwiseIncrementalCombiner,
    CentroidIncrementalCombiner,
)
from am_combiner.features.article import Features


def test_entity_cluster(article):
    entity_cluster = EntityCluster(
        [article("Some name", "Some text")],
        [np.array([1.0, 2.0, 3.0]).reshape(1, -1)],
        np.array([1.0, 2.0, 3.0]).reshape(1, -1),
    )
    assert_entity_cluster_points(entity_cluster.points, [np.array([1.0, 2.0, 3.0]).reshape(1, -1)])
    assert np.alltrue(entity_cluster.centroid == np.array([1.0, 2.0, 3.0]).reshape(1, -1))

    entity_cluster.add_new_entity(
        article("Some other name", "Some other text"),
        np.array([3.0, 4.0, 5.0]).reshape(1, -1),
    )
    assert_entity_cluster_points(
        entity_cluster.points,
        [np.array([1.0, 2.0, 3.0]).reshape(1, -1), np.array([3.0, 4.0, 5.0]).reshape(1, -1)],
    )
    assert np.alltrue(entity_cluster.centroid == np.array([2.0, 3.0, 4.0]).reshape(1, -1))

    entity_cluster.add_new_entity(
        article("Some another name", "Some another text"),
        np.array([14.6, 99.18, 5.32]).reshape(1, -1),
    )
    assert_entity_cluster_points(
        entity_cluster.points,
        [
            np.array([1.0, 2.0, 3.0]).reshape(1, -1),
            np.array([3.0, 4.0, 5.0]).reshape(1, -1),
            np.array([14.6, 99.18, 5.32]).reshape(1, -1),
        ],
    )
    assert np.alltrue(entity_cluster.centroid == np.array([6.2, 35.06, 4.44]).reshape(1, -1))


@pytest.mark.parametrize(
    "incremental_combiner, expected_highest_similarity, expected_closest_cluster",
    [
        (
            PairwiseIncrementalCombiner(Features.FULL_TEXT_FEATURES, 0.9),
            0.97,
            EntityCluster(
                [],
                [np.array([1, 2, 3]).reshape(1, -1), np.array([2, 0, 0]).reshape(1, -1)],
                np.array([1.5, 1, 1.5]).reshape(1, -1),
            ),
        ),
        (
            CentroidIncrementalCombiner(Features.FULL_TEXT_FEATURES, 0.9),
            0.90,
            EntityCluster(
                [],
                [np.array([7, 8, 9]).reshape(1, -1), np.array([10, 11, 12]).reshape(1, -1)],
                np.array([8.5, 9.5, 10.5]).reshape(1, -1),
            ),
        ),
    ],
)
def test_get_closest_cluster(
    article, incremental_combiner, expected_highest_similarity, expected_closest_cluster
):
    clusters = [
        EntityCluster(
            [],
            [np.array([1, 2, 3]).reshape(1, -1), np.array([2, 0, 0]).reshape(1, -1)],
            np.array([1.5, 1, 1.5]).reshape(1, -1),
        ),
        EntityCluster(
            [],
            [np.array([7, 8, 9]).reshape(1, -1), np.array([10, 11, 12]).reshape(1, -1)],
            np.array([8.5, 9.5, 10.5]).reshape(1, -1),
        ),
        EntityCluster(
            [],
            [np.array([13, 14, 15]).reshape(1, -1), np.array([16, 17, 18]).reshape(1, -1)],
            np.array([14.5, 15.5, 16.5]).reshape(1, -1),
        ),
    ]
    new_features = np.array([5.2, 6.18, 17.9]).reshape(1, -1)

    actual_highest_similarity, actual_closest_cluster = incremental_combiner.get_closest_cluster(
        clusters, new_features
    )

    assert expected_highest_similarity == round(actual_highest_similarity, 2)
    assert_entity_cluster_points(expected_closest_cluster.points, actual_closest_cluster.points)
    assert np.alltrue(expected_closest_cluster.centroid == actual_closest_cluster.centroid)


def assert_entity_cluster_points(expected_closest_cluster_points, actual_closest_cluster_points):
    assert len(expected_closest_cluster_points) == len(actual_closest_cluster_points)
    for index in range(len(expected_closest_cluster_points)):
        expected_closest_cluster_point = expected_closest_cluster_points[index]
        actual_closest_cluster_point = actual_closest_cluster_points[index]
        assert np.alltrue(expected_closest_cluster_point == actual_closest_cluster_point)
