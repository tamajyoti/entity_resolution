from typing import List, Tuple, Optional

import common.serializers
import pandas as pd
from common.ml.model import TFIDFCosineSimilarityClusterer

from am_combiner.combiners.common import Combiner
from am_combiner.features.article import Article
from am_combiner.splitters.common import Splitter


class LibTFIDFCosineSimilarityClusterer(Combiner):

    """
    A concrete implementation of a combiner class.

    Uses TFIDFCosineSimilarityClusterer from am_combiner_common library to cluster mentions.
    Should have the exact same behaviour as TFIDFCosineSimilarityCombiner.

    Attributes
    ----------
    threshold: float
        Cosine similarity threshold.
    vectoriser_uri: str
        The URI of the vectoriser model from S3.
    vectoriser_target_path: str
        The target path where to download the vectoriser.

    """

    def __init__(self, threshold: float, vectoriser_uri: str, vectoriser_target_path: str):
        super().__init__()
        self.threshold = threshold
        self.vectoriser_uri = vectoriser_uri
        self.vectoriser_target_path = vectoriser_target_path

    def combine_entities(
        self, input_entities: List[Article], splitter: Optional[Splitter] = None
    ) -> pd.DataFrame:
        """
        Combine a list of given articles into clusters.

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
        entity_name, mentions = self._create_mentions(input_entities)

        clusterer = TFIDFCosineSimilarityClusterer(
            self.threshold, self.vectoriser_uri, self.vectoriser_target_path
        )
        clusters = clusterer.cluster(entity_name, mentions)

        return self._output_clusters(input_entities, clusters)

    @staticmethod
    def _create_mentions(
        input_entities: List[Article],
    ) -> Tuple[str, List[common.serializers.MLMention]]:
        """Create Mention objects from Article objects."""
        entity_name = input_entities[0].entity_name

        mentions = list()
        for entity_index in range(len(input_entities)):
            mention = common.serializers.MLMention(
                mention_id=str(entity_index),
                article_text=input_entities[entity_index].article_text,
            )
            mentions.append(mention)

        return entity_name, mentions

    @staticmethod
    def _output_clusters(
        input_entities: List[Article], clusters: List[common.serializers.Cluster]
    ) -> pd.DataFrame:
        """Output clusters in dataframe format."""
        clusters_ids = list()
        cluster_urls = list()
        entity_names = list()
        for cluster_index in range(len(clusters)):
            cluster = clusters[cluster_index]

            for mention_id in cluster.mentions:
                input_entity = input_entities[int(mention_id)]

                clusters_ids.append(cluster_index)
                cluster_urls.append(input_entity.url)
                entity_names.append(input_entity.entity_name)

        return Combiner.return_output_dataframe(
            cluster_ids=clusters_ids,
            unique_ids=cluster_urls,
            blocking_names=entity_names,
        )
