import abc
from collections import defaultdict
from typing import Dict, List

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix

from am_combiner.combiners.common import (
    CombinedObject,
    BLOCKING_FIELD_FIELD,
    UNIQUE_ID_FIELD,
    CLUSTER_ID_FIELD,
    Combiner,
)
from am_combiner.combiners.graph_based import ConnectedComponentsCombiner
from am_combiner.utils.adjacency import get_article_multi_feature_adjacency


class AbstractBlocker(abc.ABC):

    """Any descendants are responsible for data blocking."""

    def __init__(self) -> None:
        self.deblock_mapping = {}

    @abc.abstractmethod
    def block_data(self, input_data: Dict[str, List[CombinedObject]]):
        """Take input data and block it into pieces."""

    def deblock_labels(self, clustering_results: pd.DataFrame) -> pd.DataFrame:
        """Map blocks names back to original. QA relies on them."""
        """
        The main problem with data de-blocking is that every group of objects, e.g.
        person, vessel, etc starts counting its cluster ids from 0. Therefore, we
        can not just merge them into the block, because you will get in-valid clusters
        not taking global ids into account. It is therefore important to track global cluster
        ids as well.
        """

        # Build composite ID and map it to a code, the code is the final cid
        clustering_results[CLUSTER_ID_FIELD] = (
            clustering_results.apply(
                lambda row: f"{row[BLOCKING_FIELD_FIELD]}-{row[CLUSTER_ID_FIELD]}", axis=1
            )
            .astype("category")
            .cat.codes
        )

        clustering_results[BLOCKING_FIELD_FIELD] = clustering_results[BLOCKING_FIELD_FIELD].apply(
            lambda block_name: self.deblock_mapping[block_name]
        )
        return clustering_results


class IdentityBlocker(AbstractBlocker):

    """No-blocker essentially. Returns the data as is."""

    def block_data(
        self, input_data: Dict[str, List[CombinedObject]]
    ) -> Dict[str, List[CombinedObject]]:
        """Return the data as is."""
        # So we just return data as a single block, no blocking happens.
        for k in input_data.keys():
            self.deblock_mapping[k] = k
        return input_data


class FeatureBasedNameBlocker(AbstractBlocker):

    """Class blocks data based on a given set of fields."""

    def __init__(self, use_features) -> None:
        super().__init__()
        combiner = ConnectedComponentsCombiner(use_features=use_features)

        self.combiner = combiner

    def block_data(
        self, input_data: Dict[str, List[CombinedObject]]
    ) -> Dict[str, List[CombinedObject]]:
        """Block the data based on the set list of fields."""
        output = defaultdict(list)
        for object_type, records in input_data.items():
            combining_results = self.combiner.combine_entities(records)
            combining_results.set_index([UNIQUE_ID_FIELD], inplace=True)
            combining_results["block_name"] = combining_results.apply(
                lambda row: f"{row[BLOCKING_FIELD_FIELD]}-{row[CLUSTER_ID_FIELD]}", axis=1
            )
            # Put each row into its bucket, while saving reverse mapping data
            for r in records:
                bn = combining_results.loc[r.sanction_id].block_name
                output[bn].append(r)
                self.deblock_mapping[bn] = object_type
                r.type = bn
        return output


class FeatureBasedNameBlockerWithCutoff(AbstractBlocker):

    """Class blocks data gradually using clustering cutoff."""

    def __init__(self, use_features, cluster_cutoff: int, th_ls: List[int]) -> None:
        super().__init__()
        assert len(use_features) == len(th_ls), "feature and threshold numbers should match."
        self.use_features = use_features
        self.cluster_cutoff = cluster_cutoff
        self.th_ls = th_ls
        self.block_num = 0
        self.output = defaultdict(list)

    def _block_over_cutoff_clusters(
        self, cluster_ids: List[int], id_to_record: Dict[str, CombinedObject], object_type: str
    ) -> Dict[str, CombinedObject]:
        """
        For clusters over cluster_cutoff threshold, perform blocking and removed blocked records.

        Parameters
        ----------
        cluster_ids:
            resulting clustering
        id_to_record:
            map from unique_id to object for unblocked records.
        object_type:
            type of objects being blocked, eg: "person", "vessel", etc.

        Returns
        -------
            updated id_to_record that no longer contains records that got assigned blocks.

        """
        clustering_df = pd.DataFrame(
            data={CLUSTER_ID_FIELD: cluster_ids, UNIQUE_ID_FIELD: id_to_record.keys()}
        )
        grouped_df = clustering_df.groupby(by=CLUSTER_ID_FIELD).count()
        clusters_over_limit = grouped_df[grouped_df[UNIQUE_ID_FIELD] >= self.cluster_cutoff]

        # Put each row into its bucket, while saving reverse mapping data
        for cluster in clusters_over_limit.index:
            bn = object_type + "-" + str(self.block_num)
            record_ids = clustering_df[clustering_df[CLUSTER_ID_FIELD] == cluster][UNIQUE_ID_FIELD]
            for record_id in record_ids:
                record = id_to_record[record_id]
                record.type = bn

                self.output[bn].append(record)
                self.deblock_mapping[bn] = object_type

                # remove already blocked records from further blocking:
                del id_to_record[record_id]
            self.block_num += 1

        return id_to_record

    @staticmethod
    def _get_adj_matrix(adj_matrices: List[coo_matrix], th: float) -> coo_matrix:
        """Create common adj matrix limiting the impact from the last feature."""
        # Apply threshold on adjacency matrix of the last feature.
        adj_matrices[-1].data[adj_matrices[-1].data < th] = 0
        adj_matrices[-1].eliminate_zeros()

        # Sum all features:
        adjacency_matrix = np.array(adj_matrices).sum(axis=0)
        adjacency_matrix = coo_matrix(adjacency_matrix)

        return adjacency_matrix

    def block_data(
        self, input_data: Dict[str, List[CombinedObject]]
    ) -> Dict[str, List[CombinedObject]]:
        """Block the data based on the set list of fields."""
        for object_type, records in input_data.items():

            id_to_record = {r.sanction_id: r for r in records}

            # Try blocking with less features first and remove blocks over the cut off.
            for feat_num in range(len(self.use_features)):
                # Try blocking with higher threshold values first.
                for th in range(self.th_ls[feat_num], 0, -1):

                    # Consider only records that weren't already assigned a block.
                    unblocked_records = list(id_to_record.values())
                    adj_matrices = get_article_multi_feature_adjacency(
                        unblocked_records, self.use_features[: (feat_num + 1)], as_list=True
                    )

                    adjacency_matrix = self._get_adj_matrix(adj_matrices, th)
                    cluster_ids = Combiner.compute_cluster_ids_from_adjacency_matrix(
                        adjacency_matrix, unblocked_records, splitter=None
                    )

                    # For the last iteration, block all remaining records:
                    if feat_num == len(self.use_features) - 1 and th == 1:
                        self.cluster_cutoff = 0

                    # Remove records that are already blocked.
                    id_to_record = self._block_over_cutoff_clusters(
                        cluster_ids, id_to_record, object_type
                    )
        return self.output
