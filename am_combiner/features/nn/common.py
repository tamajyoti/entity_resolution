from typing import Union, Tuple, List

import dgl
import networkx as nx
import numpy as np
import scipy
import torch

from am_combiner.utils.adjacency import get_article_multi_feature_adjacency
from am_combiner.features.article import Features, Article
from scipy.sparse import coo_matrix

HETEROGENEOUS_NODE_NAME = "article"


class UnknownNumericFeatureTypeException(Exception):

    """Exception for when we do not know how to convert a feature."""

    pass


def get_node_embeddings(
    input_entities: List[Article],
    node_feature: Union[Features, str],
):
    """

    Select article feature to be used as node embedding.

    Parameters
    ----------
    input_entities:
        List of articles to build the graph for
    node_feature:
        Name of the Article features to use as node embeddings

    Returns
    -------
        node embeddings as torch parameters

    """
    features = []
    for article in input_entities:
        this_feature = article.extracted_entities.get(node_feature, None)
        if this_feature is None:
            raise KeyError("No such feature exists for node embedding.")
        if isinstance(this_feature, np.ndarray):
            pass
        elif isinstance(this_feature, scipy.sparse.csr.csr_matrix) and this_feature.shape[0] == 1:
            # Transform to dense
            this_feature = this_feature.A[0]
        else:
            raise UnknownNumericFeatureTypeException(
                f"Don't know how to convert to dense: {type(this_feature)}"
            )

        features.append(this_feature)
    features = np.array(features)

    # Create an un-trainable embedding objects
    number_of_embeddings, embeddings_dimensions = features.shape
    embedding = torch.nn.Embedding(number_of_embeddings, embeddings_dimensions)
    embedding.weight = torch.nn.Parameter(torch.from_numpy(features).float(), requires_grad=False)

    return embedding


def articles_to_homogeneous_graph(
    input_entities: List[Article],
    node_feature: Union[Features, str],
    edge_features: Tuple[Union[Features, str], ...],
):
    """

    Transform a list of articles into a homogeneous graph.

    Parameters
    ----------
    input_entities:
        List of articles to build the graph for
    node_feature:
        Name of the Article features to use as node embeddings
    edge_features:
        List of features that define connectivity

    Returns
    -------
    Tuple[DGLGraph, torch.Parameter]
        DGLGraph and it's nodes embeddings

    """
    adjacency_matrix = get_article_multi_feature_adjacency(
        input_entities, edge_features, as_list=False
    )

    # hack to avoid empty graph error:
    if adjacency_matrix.data.sum() == 0:
        adjacency_matrix = coo_matrix(
            ([1], ([0], [0])), shape=adjacency_matrix.shape, dtype=np.int32
        )

    Gnx = nx.DiGraph(adjacency_matrix)
    G = dgl.from_networkx(Gnx, edge_attrs=["weight"])

    embedding = get_node_embeddings(input_entities, node_feature)

    return G, embedding.weight


def articles_to_hetero_graph(
    input_entities: List[Article],
    node_feature: Union[Features, str],
    edge_features: Tuple[Union[Features, str], ...],
):
    """
    Transform articles into training samples for multigraph training.

    Parameters
    ----------
    input_entities:
        List of articles to build the graph for
    node_feature:
        Name of the Article features to use as node embeddings
    edge_features:
        List of features that define connectivity

    Returns
    -------
    Tuple[DGLGraph, torch.Parameter]
        DGLGraph and it's nodes embeddings

    """
    # Create a feature adjacency matrix and build a graph from it
    adjacency_matrices = get_article_multi_feature_adjacency(
        input_entities, edge_features, as_list=True
    )

    # hack to avoid empty graph error:
    edge_nums = [len(ad.data) for ad in adjacency_matrices]
    if max(edge_nums) == 0:
        adjacency_matrices[0] = coo_matrix(
            ([1], ([0], [0])), shape=adjacency_matrices[0].shape, dtype=np.int32
        )

    multigaph_source = {}

    for adjacency_matrix, edge_feature in zip(adjacency_matrices, edge_features):
        triplet = (
            HETEROGENEOUS_NODE_NAME,
            str(edge_feature).split(".")[-1],
            HETEROGENEOUS_NODE_NAME,
        )
        multigaph_source[triplet] = (
            torch.tensor(adjacency_matrix.col).to(torch.int32),
            torch.tensor(adjacency_matrix.row).to(torch.int32),
        )

    G = dgl.heterograph(
        multigaph_source, num_nodes_dict={HETEROGENEOUS_NODE_NAME: len(input_entities)}
    )

    for adjacency_matrix, edge_feature in zip(adjacency_matrices, edge_features):
        G.edges[str(edge_feature).split(".")[-1]].data["weight"] = torch.Tensor(
            adjacency_matrix.data
        )

    embedding = get_node_embeddings(input_entities, node_feature)

    return G, {HETEROGENEOUS_NODE_NAME: embedding.weight}
