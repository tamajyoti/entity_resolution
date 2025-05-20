from typing import List

from am_combiner.combiners.common import Combiner
from am_combiner.features.article import Article


class CurrentProductionCombiner(Combiner):

    """
    A concrete implementation of an abstract class representing a generic Combiner.

    This one implements the current functionality of the production combiner.
    It therefore does no job. It only assigns every entity the same cluster id.

    """

    def combine_entities(self, input_entities: List[Article], splitter=None):
        """
        Combine a list of given articles into clusters.

        Parameters
        ----------
        input_entities:
            A list of Article objects to be combined.
        splitter:
            Splitter does not apply in this case.

        Returns
        -------
            A pd.DataFrame object with cluster ids assigned to all articles.

        """
        # Basically everything with the same name goes into the same cluster
        return Combiner.return_output_dataframe(
            cluster_ids=[0] * len(input_entities),
            unique_ids=[article.url for article in input_entities],
            blocking_names=[article.entity_name for article in input_entities],
        )
