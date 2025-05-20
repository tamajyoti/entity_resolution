from typing import List, Union, Optional
import numpy as np
import pandas as pd

from am_combiner.combiners.common import Combiner
from am_combiner.features.article import Article, Features
from am_combiner.combiners.tfidf import TFIDFAndFeaturesCosineSimilarityCombiner
from am_combiner.splitters.common import Splitter


class AnnotationsCombiner(TFIDFAndFeaturesCosineSimilarityCombiner):

    """
    A concrete implementation of a combiner class.

    Overwrites TFIDFAndFeaturesCosineSimilarityCombiner edges with
    annotated data.

    TFIDFAndFeaturesCosineSimilarityCombiner combiner is chosen for two reasons:
    1) It has adjacency matrix to be over-writen with annotation scores.
    2) This combiner has been used to decide similarity scores for annotation selection.

    In practice, best performing combiner with adjacency_matrix object should be used.
    """

    def __init__(
        self,
        use_features: List[Features],
        th: float = 0.5,
        max_energy: int = 75,
        source_feature: Union[Features, str] = Features.TFIDF_FULL_TEXT,
        mongo_uri: Optional[str] = None,
        mongo_collection: Optional[str] = None,
    ):
        super().__init__(th, source_feature, mongo_uri, mongo_collection)
        self.max_energy = max_energy
        self.use_features = use_features
        self.source_feature = source_feature
        self.th = th

    @staticmethod
    def _overwrite_features_with_annotations(
        adjacency_matrix: np.array(float), input_entities: List[Article]
    ) -> np.array(float):

        url_to_index = {}
        for inx, article in enumerate(input_entities):
            url_to_index[article.url] = inx

        for article in input_entities:

            # For each positive annotation, overwrite adjacency matrix entry to 1:
            for annotated_url in article.positive_urls:

                i = url_to_index[article.url]
                j = url_to_index[annotated_url]

                adjacency_matrix[i, j] = 1

            # For each negative annotation, overwrite adjacency matrix entry to 0:
            for annotated_url in article.negative_urls:
                i = url_to_index[article.url]
                j = url_to_index[annotated_url]

                adjacency_matrix[i, j] = 0

        return adjacency_matrix

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
            Splitter on resulting adjacency matrix.

        Returns
        -------
            A pd.DataFrame object with cluster ids assigned to all articles.

        """
        sim = self._get_pairwise_similarities(input_entities)
        sim = self._enhance_pairwise_similarities(sim, input_entities)
        adjacency_matrix = self._get_adjacency_from_similarities(sim)
        adjacency_matrix = self._overwrite_features_with_annotations(
            adjacency_matrix, input_entities
        )
        cluster_ids = Combiner.compute_cluster_ids_from_adjacency_matrix(
            adjacency_matrix, input_entities, splitter
        )
        return Combiner.return_output_dataframe(
            cluster_ids=cluster_ids,
            unique_ids=[article.url for article in input_entities],
            blocking_names=[article.entity_name for article in input_entities],
        )
