from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from am_combiner.combiners.common import Combiner
from am_combiner.features.article import Article, Features
from am_combiner.splitters.common import Splitter


@dataclass
class EntityCluster:

    """Helper class to encapsulate a cluster of entities."""

    entities: List[Article]
    points: List[np.array]
    centroid: np.array

    def add_new_entity(self, new_entity: Article, new_point: np.array):
        """Add a new entity to the cluster and update the centroid."""
        self.entities.append(new_entity)
        self.points.append(new_point)
        self.centroid = np.sum(self.points, axis=0) / len(self.points)


class IncrementalCombiner(Combiner):

    """
    An implementation of a combiner class.

    Simulates incremental clustering, goes through entities one by one and assigns each entity to
    the closest cluster, which exceeds a threshold.

    If no cluster exceeds the threshold, create a new cluster.

    Attributes
    ----------
    source_feature: Features
        The feature to use (can be TFIDF or BERT).
    threshold: float
        Cosine similarity threshold.

    """

    def __init__(self, source_feature: Features, threshold: float):
        super().__init__()

        self.source_feature = source_feature
        self.threshold = threshold

    def combine_entities(
        self, input_entities: List[Article], splitter: Optional[Splitter] = None
    ) -> pd.DataFrame:
        """
        Combine a list of given articles into clusters, using an incremental approach.

        Parameters
        ----------
        input_entities:
            A list of Article objects to be combined.
        splitter:
            Splitter can't be applied here due to lack of adjacency matrix.

        Returns
        -------
            A pd.DataFrame object with cluster ids assigned to all articles.

        """
        clusters: List[EntityCluster] = list()

        for new_entity in input_entities:
            new_features = new_entity.extracted_entities[self.source_feature]

            highest_similarity, closest_cluster = self.get_closest_cluster(clusters, new_features)

            if highest_similarity >= self.threshold and closest_cluster:
                closest_cluster.add_new_entity(new_entity, new_features)
            else:
                clusters.append(EntityCluster([new_entity], [new_features], new_features))

        return self.output_clusters(clusters)

    @abstractmethod
    def get_closest_cluster(
        self, clusters: List[EntityCluster], new_features: np.array
    ) -> Tuple[float, Optional[EntityCluster]]:
        """Get closest cluster for a new entity, given a list of clusters."""
        pass

    @staticmethod
    def output_clusters(clusters: List[EntityCluster]):
        """Output the clustering results in a standard format."""
        cluster_ids, urls, names = list(), list(), list()
        for cluster_index in range(len(clusters)):
            cluster = clusters[cluster_index]
            for entity in cluster.entities:
                cluster_ids.append(cluster_index)
                urls.append(entity.url)
                names.append(entity.entity_name)

        return Combiner.return_output_dataframe(
            cluster_ids=cluster_ids,
            unique_ids=urls,
            blocking_names=names,
        )


class PairwiseIncrementalCombiner(IncrementalCombiner):

    """
    A concrete implementation of an incremental combiner class.

    Simulates incremental clustering, goes through entities one by one and assigns each entity to
    the closest cluster.

    The closest cluster is decided by calculating the cosine similarity with all points from all
    clusters and assigning the new entity to the cluster which contains the element with the
    highest similarity that exceeds a threshold.

    If no cluster exceeds the threshold, create a new cluster.

    """

    def get_closest_cluster(
        self, clusters: List[EntityCluster], new_features: np.array
    ) -> Tuple[float, Optional[EntityCluster]]:
        """
        Get closest cluster of a new entity, given a list of clusters.

        The closest cluster is decided by calculating the cosine similarity of the new entity with
        all entities from all clusters.

        Parameters
        ----------
        clusters:
            A list of clusters to get closest cluster from.
        new_features:
            The features of the new entity to get closest cluster for.

        Returns
        -------
            A tuple of the highest similarity and the corresponding closest cluster.

        """
        highest_similarity = 0.0
        closest_cluster = None

        for cluster in clusters:
            for features in cluster.points:
                similarity = cosine_similarity(new_features, features)[0][0]

                if similarity > highest_similarity:
                    highest_similarity = similarity
                    closest_cluster = cluster

        return highest_similarity, closest_cluster


class CentroidIncrementalCombiner(IncrementalCombiner):

    """
    A concrete implementation of an incremental combiner class.

    Simulates incremental clustering, goes through entities one by one and assigns each entity to
    the closest cluster.

    The closest cluster is decided by calculating the cosine similarity with each cluster centroid
    and assigning the new entity to the cluster which contains the centroid with the highest
    similarity that exceeds a threshold.

    If no cluster exceeds the threshold, create a new cluster.

    """

    def get_closest_cluster(
        self, clusters: List[EntityCluster], new_features: np.array
    ) -> Tuple[float, Optional[EntityCluster]]:
        """
        Get closest cluster of a new entity, given a list of clusters.

        The closest cluster is decided by calculating the cosine similarity of the new entity with
        the centroid of each cluster.

        Parameters
        ----------
        clusters:
            A list of clusters to find closest cluster from.
        new_features:
            The features of the new entity to find closest cluster for.

        Returns
        -------
            A tuple of the highest similarity and the corresponding closest cluster.

        """
        closest_cluster = None
        highest_similarity = 0.0

        for cluster in clusters:
            similarity = cosine_similarity(new_features, cluster.centroid)[0][0]

            if similarity > highest_similarity:
                highest_similarity = similarity
                closest_cluster = cluster

        return highest_similarity, closest_cluster
