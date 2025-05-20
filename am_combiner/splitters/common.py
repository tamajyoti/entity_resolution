import abc
from typing import List, Optional, Union
from scipy.sparse import coo_matrix
import numpy as np
import networkx as nx

from am_combiner.features.article import Article, Features
from am_combiner.utils.adjacency import get_multi_feature_negative_edges
from am_combiner.features.sanction import Sanction, SanctionFeatures
from am_combiner.utils.parametrization import features_str_to_enum, get_cache_from_yaml

CombinedObject = Union[Article, Sanction]
CombinedObjectFeature = Union[Features, SanctionFeatures]


class Splitter(abc.ABC):

    """Splitter base class."""

    def __init__(
        self,
        negative_features: List[CombinedObjectFeature],
        numeric_distances: List[Optional[int]],
    ):
        self.negative_features = negative_features
        self.numeric_distances = numeric_distances
        self.cluster_counter = None
        self.cluster_ids = None
        self.negators = []

        assert len(negative_features) == len(
            numeric_distances
        ), "Negative feature number does not match distances inputted."
        for i, f in enumerate(self.negative_features):
            self.negators.append((f, self.numeric_distances[i]))

    @abc.abstractmethod
    def break_subgraph(self, pos_adj: np.array, neg_adj: np.array, ids: List[int]):
        """Use positive and negative adjacency matrices to break up a subgraph."""
        pass

    def get_sliced_adjacencies(
        self, ids: List[int], entities: List[CombinedObject], adj_matrix: np.array
    ) -> None:
        """
        Break apart existent subgraph.

        Parameters
        ----------
        ids:
            list of node indexes relating to subgraph.
        entities:
            entities corresponding to subgraph nodes.
        adj_matrix:
            entire adjacency matrix used to derive subgraphs.

        """
        neg_adj = get_multi_feature_negative_edges(entities, negators=self.negators)

        # No breaking happens without negative edges:
        if neg_adj.sum() == 0:
            for node in ids:
                self.cluster_ids[node] = self.cluster_counter
            self.cluster_counter += 1
            return

        pos_adj = self.slice_pos_adjacency(adj_matrix, ids)
        self.break_subgraph(pos_adj.toarray(), neg_adj, ids)

    @staticmethod
    def slice_pos_adjacency(adj_matrix: coo_matrix, blocked_ids: List[int]) -> coo_matrix:
        """
        Slicing global adjacency matrix to contain only subgraph edges.

        Parameters
        ----------
        adj_matrix:
            global adjacency matrix in coo_matrix format.
        blocked_ids:
            list of subgraph node ids.

        Returns
        -------
            coo_matrix that correspond to subgraph adjacency.

        """
        n = len(blocked_ids)

        mask_row = np.isin(adj_matrix.row, blocked_ids)
        mask_col = np.isin(adj_matrix.col, blocked_ids)
        mask_all = np.logical_and(mask_col, mask_row)

        sliced_row = adj_matrix.row[mask_all]
        sliced_col = adj_matrix.col[mask_all]
        sliced_data = adj_matrix.data[mask_all]

        # remunerate row and columns indexes:
        blocked_map = {}
        for inx, blocked_id in enumerate(blocked_ids):
            blocked_map[blocked_id] = inx
        sliced_row = [blocked_map[i] for i in sliced_row]
        sliced_col = [blocked_map[i] for i in sliced_col]

        sliced_adj = coo_matrix(
            (sliced_data, (sliced_row, sliced_col)), shape=(n, n), dtype=np.int32
        )
        return sliced_adj

    def split(
        self, cluster_ids: List[int], adj_matrix: coo_matrix, input_entities: List[CombinedObject]
    ) -> List[int]:
        """For each cluster, invoke braking methods."""
        assert isinstance(
            adj_matrix, coo_matrix
        ), f"Adjacency matrix expected in coo_matrix format, got {type(adj_matrix)}"
        n = len(cluster_ids)
        old_cluster_ids = np.array(cluster_ids)

        self.cluster_ids = [0] * n
        self.cluster_counter = 0

        for cluster in set(old_cluster_ids):
            blocked_ids = list(np.arange(n)[old_cluster_ids == cluster])
            blocked_ents = [input_entities[i] for i in blocked_ids]
            self.get_sliced_adjacencies(blocked_ids, blocked_ents, adj_matrix)

        return self.cluster_ids


class ColourNegativeEdgesSplitter(Splitter):

    """Use colouring algorithm to enforce subgraph with negative edges being split."""

    def __init__(
        self,
        negative_features: List[CombinedObjectFeature],
        numeric_distances: List[Optional[int]],
    ):
        super().__init__(negative_features, numeric_distances)

    def break_subgraph(self, pos_adj: np.array, neg_adj: np.array, ids: List[int]) -> None:
        """

        Update cluster_ids enforcing negative edges to never appear in the same cluster.

        Parameters
        ----------
        pos_adj:
            positive adjacency matrix of subgraph.
        neg_adj:
            negative adjacency matrix of subgraph.
        ids:
            global ids of subgraph's nodes.

        """
        neg_graph = nx.Graph(neg_adj)
        n = neg_graph.number_of_nodes()
        colouring = nx.coloring.greedy_color(neg_graph, strategy="largest_first")

        # Nodes with at least one negative edge, get assigned the colour cluster.
        final_coloring = -np.ones(n)
        for node, colour in colouring.items():
            if neg_graph.degree(node) > 0:
                final_coloring[node] = colour

        # Deal with isolated edges: assign to cluster with highest mean edge:
        for node in colouring:
            if neg_graph.degree(node) == 0:
                mean_edges = []
                for colour in range(max(colouring.values()) + 1):
                    pos_edges = pos_adj[node, final_coloring == colour]
                    mean_edges.append(pos_edges.mean())
                final_coloring[node] = np.argmax(mean_edges)

        # Convert local nodes & colours to global cluster ids:
        for node in range(n):
            self.cluster_ids[ids[node]] = self.cluster_counter + final_coloring[node]
        self.cluster_counter += max(colouring.values()) + 1


class DeleteNegativeEdgesSplitter(Splitter):

    """Delete negative edges from a subgraph."""

    def __init__(
        self,
        negative_features: List[CombinedObjectFeature],
        numeric_distances: List[Optional[int]],
    ):
        super().__init__(negative_features, numeric_distances)

    def break_subgraph(self, pos_adj: np.array, neg_adj: np.array, ids: List[int]) -> None:
        """

        Delete negative edges from a subgraph.

        Parameters
        ----------
        pos_adj:
            positive adjacency matrix of subgraph.
        neg_adj:
            negative adjacency matrix of subgraph.
        ids:
            global ids of subgraph's nodes.

        """
        neg_adj = np.clip(neg_adj, 0, 1)
        pos_adj = np.clip(pos_adj, 0, 1)
        final_adj = np.clip(pos_adj - neg_adj, 0, 1)

        reduced_graph = nx.Graph(final_adj)
        sub_subgraphs = nx.connected_components(reduced_graph)
        for sub_subgraph in sub_subgraphs:
            for entity_id in sub_subgraph:
                self.cluster_ids[ids[entity_id]] = self.cluster_counter
            self.cluster_counter += 1


def load_splitter(splitter: Splitter, config_path: str) -> Optional[Splitter]:
    """Load splitter object."""
    splitter = get_cache_from_yaml(
        config_path,
        section_name="splitters",
        class_mapping=SPLITTER_CLASS_MAPPING,
        restrict_classes={splitter},
        attrs_callbacks={"negative_features": lambda fs: [features_str_to_enum(f) for f in fs]},
    )
    if splitter:
        return list(splitter.values())[0]
    return


SPLITTER_CLASS_MAPPING = {
    "DeleteNegativeEdgesSplitter": DeleteNegativeEdgesSplitter,
    "ColourNegativeEdgesSplitter": ColourNegativeEdgesSplitter,
}
