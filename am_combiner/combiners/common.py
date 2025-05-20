import abc
from collections import defaultdict
from typing import List, Tuple, Dict, Union, Optional
from tqdm import tqdm
import networkx as nx
import time

import numpy as np
import pandas as pd

from am_combiner.features.article import Article, Features
from am_combiner.features.sanction import Sanction, SanctionFeatures
from am_combiner.splitters.common import Splitter

ID_FIELD = "_id"
ENTITY_NAME_FIELD = "entity_name"
META_DATA_FIELD = "meta_data"
TEXT_COLUMN_FIELD = "content"
URL_FIELD = "url"
CLUSTER_ID_FIELD = "ClusterID"
CLUSTER_ID_GLOBAL_FIELD = "ClusterIDGlobal"

# Validation fields:
GROUND_TRUTH_FIELD = "ground_truth"
BLOCKING_FIELD_FIELD = "blocking_field"
UNIQUE_ID_FIELD = "unique_id"

# Sanction fields:
SANCTION_ID_FIELD = "sanction_id"
SANCTION_TYPE_FIELD = "entity_type"
PROFILE_ID_FIELD = "profile_id"
SANCTION_ENTITY_FIELD = "entity"
TRAIN_TEST_VALIDATE_SPLIT_FIELD = "split"

# Field names for quality evaluation
IS_OVER_FIELD = "is over"
IS_UNDER_FIELD = "is under"
CLUSTER_SUPPORT_FIELD = "cluster_support"
ENTITY_NAME_CLUSTER_ID_FIELD = "entity_name_cluster_id"
CLUSTER_NUMBER_FIELD = "ClusterID"

NAME_FIELD = "Name"
HOMOGENEITY_FIELD = "Homogeneity"
COMPLETENESS_FIELD = "Completeness"
V_SCORE_FIELD = "V score"
COUNT_FIELD = "Count"
NAME_OC_RATE_FIELD = "Name OC rate"
NAME_UC_RATE_FIELD = "Name UC rate"
PROFILES_PER_OC_FIELD = "Profiles per OC"
PROFILES_CREATED_FIELD = "Profiles created"
PROFILES_TRUE_FIELD = "True profiles"
SCORE_TO_MINIMISE_FIELD = "Score to minimize"

CombinedObject = Union[Article, Sanction]
CombinedObjectFeatureStr = Union[Features, SanctionFeatures, str]
CombinedObjectFeature = Union[Features, SanctionFeatures]


def load_combiner_input_csv(csv_path: str, ignore_missing_cols: bool = False) -> pd.DataFrame:
    """
    Load a csv and check whether all the required fields are there.

    This function is used exclusively for loading combiner input input csv, hence
    the enforced structure of the fields required.

    Parameters
    ----------
    csv_path:
        A path to a csv file to be loaded.
    ignore_missing_cols:
        A flag indicating whether missing fields should be ignored.

    Returns
    -------
        A pd.DataFrame object.

    """
    loaded_dataframe = pd.read_csv(csv_path)

    if not ignore_missing_cols:
        this_cols = loaded_dataframe.columns
        for c in [BLOCKING_FIELD_FIELD, TEXT_COLUMN_FIELD, UNIQUE_ID_FIELD, GROUND_TRUTH_FIELD]:
            if c not in this_cols:
                raise ValueError(f'Required column "{c}" is not found in dataframe {csv_path}')

    # Removes empty text values
    loaded_dataframe = loaded_dataframe.dropna(subset=[TEXT_COLUMN_FIELD])

    # Removes duplicated URLs for the name name
    loaded_dataframe = loaded_dataframe.drop_duplicates([BLOCKING_FIELD_FIELD, UNIQUE_ID_FIELD])

    return loaded_dataframe


class Combiner(abc.ABC):

    """Combiner base class."""

    _use_features: List[Features]

    def __init__(self):
        self._use_features = []

    @property
    def use_features(self):
        """Getter for use features attribute."""
        return self._use_features

    @use_features.setter
    def use_features(self, new_features):
        """Setter for use features attribute."""
        self._use_features = new_features

    @abc.abstractmethod
    def combine_entities(
        self, input_entities: List[Article], splitter: Optional[Splitter] = None
    ) -> pd.DataFrame:
        """
        Combine a list of given articles into some number of clusters.

        In order to be compatible with the rest of the pipeline it is advised to
        use self.return_output_dataframe to get the result properly formatted.

        Parameters
        ----------
        input_entities:
            A list of Article objects to be combined.
        splitter:
            For combiners with final adjacency matrix, allow splitter objects.

        Returns
        -------
            A pd.DataFrame object with cluster ids assigned to all articles.

        """
        pass

    @staticmethod
    def compute_cluster_ids_from_adjacency_matrix(
        adjacency_matrix: np.array,
        input_entities: List[Article],
        splitter: Optional[Splitter],
    ) -> List[int]:
        """
        Compute cluster ids using connected components approach from a given binary matrix.

        Parameters
        ----------
        adjacency_matrix:
            (n, n) matrix with {0, 1} entries, where
            adjacency_matrix[i, j] = 1 implies edge from node i to node j.
        input_entities:
            List of entities.
        splitter:
            If not None, apply splitter on existent results.

        Returns
        -------
            List of cluster ids

        """
        assert len(adjacency_matrix.shape) == 2, "Adjacency matrix is 2-dimensional."
        assert (
            adjacency_matrix.shape[0] == adjacency_matrix.shape[1]
        ), "Adjacency matrix must have column dim to equal row dimension - i.e. be a square."

        g = nx.Graph(adjacency_matrix)
        sub_graphs = nx.connected_components(g)
        cluster_ids = [0] * adjacency_matrix.shape[0]
        for cluster_id, subgraph in enumerate(sub_graphs):
            for entity_id in subgraph:
                cluster_ids[entity_id] = cluster_id

        if splitter is not None:
            return splitter.split(cluster_ids, adjacency_matrix, input_entities)

        return cluster_ids

    @staticmethod
    def return_output_dataframe(
        cluster_ids: List[int], unique_ids: List[str], blocking_names: List[str]
    ) -> pd.DataFrame:
        """
        Build a pandas dataframe.

        It can be consumed by the downstream quality evaluation components.

        Parameters
        ----------
        cluster_ids:
            List of cluster ids assigned to each url, i.e. [1,1,2,3,4,1].
        unique_ids:
            List of urls/ids that were clustered, i.e. ['http://msc.com','http://mvp.com',...].
        blocking_names:
            List of entity names that were clustered.
            For now we only try to combine only entities with the same name,
            the list is supposed to be filled with the same entity name,
            e.g. ['Michael Bennet','Michael Bennet','Michael Bennet'].

        Returns
        -------
            DataFrame that is directly consumable by the downstream quality evaluation components.

        """
        dfs = []
        for cid, cl, en in zip(cluster_ids, unique_ids, blocking_names):
            dfs.append(
                {
                    CLUSTER_NUMBER_FIELD: cid,
                    UNIQUE_ID_FIELD: cl,
                    BLOCKING_FIELD_FIELD: en,
                }
            )

        cluster_df = pd.DataFrame(dfs)

        return cluster_df


def combine_entities_wrapper(
    entity_articles: Dict[str, List[CombinedObject]],
    combiner_object: Combiner,
    splitter: Splitter = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Combine entities into clusters.

    Parameters
    ----------
    entity_articles:
        Entities to combine.
    combiner_object:
        A Combiner instance.
    splitter:
        A Splitter instance to utilise negative information.

    Returns
    -------
        A dataframe containing the clustering results and the average time to cluster a name.

    """
    # Dataframe holding final clustering results.
    # The logic is applied on top of it and each name is assigned a cluster
    clustering_results_dataframe = pd.DataFrame()

    average_time_by_mention_no = defaultdict(list)

    entity_names = entity_articles.keys()
    for entity_ct, entity_name in tqdm(enumerate(entity_names), total=len(entity_names)):
        start = time.time()
        clustered_entity_df = combiner_object.combine_entities(
            entity_articles[entity_name], splitter=splitter
        )
        average_time_by_mention_no[len(entity_articles[entity_name])].append(time.time() - start)

        clustering_results_dataframe = clustering_results_dataframe.append(clustered_entity_df)
    clustering_results_dataframe.reset_index(inplace=True)
    for mention_no, times in average_time_by_mention_no.items():
        average_time_by_mention_no[mention_no] = (sum(times) / len(times)) * 1000

    return clustering_results_dataframe, average_time_by_mention_no


def get_unique_ids_and_blocking_fields(input_articles: List[CombinedObject]):
    """Get blocking and ids lists, depending on entity type."""
    if input_articles and isinstance(input_articles[0], Article):
        unique_ids = [article.url for article in input_articles]
        blocking_names = [article.entity_name for article in input_articles]
    elif input_articles and isinstance(input_articles[0], Sanction):
        unique_ids = [sanction.sanction_id for sanction in input_articles]
        blocking_names = [sanction.type for sanction in input_articles]

    return unique_ids, blocking_names
