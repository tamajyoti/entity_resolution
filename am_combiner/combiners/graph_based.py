from typing import List, Optional

import numpy as np
import pandas as pd
from am_combiner.combiners.common import Combiner, CombinedObject, CombinedObjectFeature

from am_combiner.utils.adjacency import get_article_multi_feature_adjacency
from am_combiner.combiners.common import get_unique_ids_and_blocking_fields
from am_combiner.splitters.common import Splitter


class ConnectedComponentsCombiner(Combiner):

    """
    A concrete implementation of the Combiner abstract class.

    Implements a combiner that combines entities using connected components approach.

    """

    def __init__(self, use_features: List[CombinedObjectFeature], th: int = 1, splitters=None):
        super().__init__()
        self.use_features = use_features
        self.th = th
        self.splitters = splitters

    def compute_adjacency_matrix(self, input_articles: List[CombinedObject]):
        """Compute adjacency matrix."""
        adjacency_matrix = get_article_multi_feature_adjacency(
            input_articles, features=self.use_features
        )

        total_adjacency_data = adjacency_matrix.data
        adjacency_data = np.zeros(adjacency_matrix.data.shape[0])
        adjacency_data[total_adjacency_data >= self.th] = 1
        adjacency_matrix.data = adjacency_data
        adjacency_matrix.eliminate_zeros()
        return adjacency_matrix

    def combine_entities(
        self, input_articles: List[CombinedObject], splitter: Optional[Splitter] = None
    ) -> pd.DataFrame:
        """
        Combine a list of given articles into clusters.

        Parameters
        ----------
        input_articles:
            A list of Article objects to be combined.
        splitter:
            Splitter on resulting adjacency matrix.

        Returns
        -------
            A pd.DataFrame object with cluster ids assigned to all articles.

        """
        adjacency_matrix = self.compute_adjacency_matrix(input_articles)

        unique_ids, blocking_names = get_unique_ids_and_blocking_fields(input_articles)
        cluster_ids = Combiner.compute_cluster_ids_from_adjacency_matrix(
            adjacency_matrix, input_articles, splitter
        )

        return Combiner.return_output_dataframe(
            cluster_ids=cluster_ids,
            unique_ids=unique_ids,
            blocking_names=blocking_names,
        )
