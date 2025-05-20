from collections import defaultdict
from typing import Dict, List, Optional, Set

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix

from am_combiner.combiners.common import (
    CombinedObject,
    BLOCKING_FIELD_FIELD,
    UNIQUE_ID_FIELD,
    CLUSTER_ID_FIELD,
)
from am_combiner.combiners.graph_based import ConnectedComponentsCombiner
from am_combiner.blockers.common import AbstractBlocker
from am_combiner.features.sanction import SanctionFeatures


class CountryBlocker(AbstractBlocker):

    """Class blocks data based on country code."""

    COUNTRY_FEATURE = SanctionFeatures.COUNTRY_CODE
    NULL_COUNTRY = "00"

    def __init__(self, use_features, min_split_size=1000) -> None:
        super().__init__()
        self.use_features = use_features
        self.combiner = ConnectedComponentsCombiner(use_features=use_features)
        self.combining_results = pd.DataFrame()
        self.cluster_country = {}
        self.min_split_size = min_split_size

    @staticmethod
    def _calculate_best_country(
        i: int,
        ids_by_country: Dict[str, Set[int]],
        adj: coo_matrix,
        filtered_records: List[CombinedObject],
        country_options: Optional[List[str]],
    ) -> str:
        """If there is no unique country code, find to the most connected country cluster."""
        connected_edges = set(adj.row[(adj.col == i)])
        if country_options is None:
            country_options = [
                filtered_records[j].extracted_entities[CountryBlocker.COUNTRY_FEATURE]
                for j in connected_edges
            ]
            country_options = sorted(set.union(*country_options))

        # If not connected to anything-country related, then assign country-less sub-cluster:
        if not country_options:
            return CountryBlocker.NULL_COUNTRY

        mean_edge_connectivity = []
        for cc in country_options:
            if len(ids_by_country[cc]) == 0:
                mean_connectivity = 0.0
            else:
                edges_to_country = connected_edges.intersection(ids_by_country[cc])
                mean_connectivity = len(edges_to_country) / len(ids_by_country[cc])
            mean_edge_connectivity.append(mean_connectivity)

        # return most connected country:
        return country_options[np.argmax(mean_edge_connectivity)]

    def _split_cluster_by_country(self, filtered_records: List[CombinedObject]) -> None:
        """Split too large cluster by country."""
        adj = self.combiner.compute_adjacency_matrix(filtered_records)
        ccs = [r.extracted_entities[CountryBlocker.COUNTRY_FEATURE] for r in filtered_records]
        ids_by_country = defaultdict(set)
        for i, cc in enumerate(ccs):
            if len(cc) == 1:
                country = next(iter(cc))
                ids_by_country[country].add(i)

        for i, r in enumerate(filtered_records):
            # All records with unique country gets into that country sub-cluster:
            if len(r.extracted_entities[CountryBlocker.COUNTRY_FEATURE]) == 1:
                country = next(iter(ccs[i]))
            # if there is more than 1 country code, assign block with largest mean connectivity:
            elif len(r.extracted_entities[CountryBlocker.COUNTRY_FEATURE]) > 1:
                country = self._calculate_best_country(
                    i, ids_by_country, adj, filtered_records, sorted(ccs[i])
                )
            else:
                country = self._calculate_best_country(
                    i, ids_by_country, adj, filtered_records, None
                )

            self.cluster_country[r.sanction_id] = country

    def block_data(
        self, input_data: Dict[str, List[CombinedObject]]
    ) -> Dict[str, List[CombinedObject]]:
        """Block the data based on the set list of fields."""
        output = defaultdict(list)
        for object_type, records in input_data.items():

            self.combining_results = self.combiner.combine_entities(records)
            cluster_sizes = self.combining_results.groupby([CLUSTER_ID_FIELD]).count()

            # If cluster exceeds max cluster size, split it by country:
            too_large_clusters = cluster_sizes[
                cluster_sizes[BLOCKING_FIELD_FIELD] > self.min_split_size
            ].index
            for cluster in too_large_clusters:
                input_ids = self.combining_results[
                    self.combining_results[CLUSTER_ID_FIELD] == cluster
                ].index
                filtered_records = [records[i] for i in input_ids]
                self._split_cluster_by_country(filtered_records)

            self.combining_results.set_index([UNIQUE_ID_FIELD], inplace=True)
            self.combining_results["block_name"] = self.combining_results.apply(
                lambda row: f"{row[BLOCKING_FIELD_FIELD]}-{row[CLUSTER_ID_FIELD]}", axis=1
            )
            # Put each row into its bucket, while saving reverse mapping data
            for r in records:
                bn = self.combining_results.loc[r.sanction_id].block_name
                if r.sanction_id in self.cluster_country:
                    bn = f"{bn}+{self.cluster_country[r.sanction_id]}"
                output[bn].append(r)
                self.deblock_mapping[bn] = object_type
                r.type = bn
        return output
