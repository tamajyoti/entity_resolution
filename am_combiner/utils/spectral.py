from copy import copy

import networkx as nx
import numpy as np
import pandas as pd
from sklearn import metrics
from collections import Counter

from am_combiner.features.article import Article, Features
from typing import List, Any, Tuple
from collections import defaultdict


def get_graph_eign(
    input_articles: List[Article], use_features: List[Features]
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    Return graph object and eigen value and vectors.

    :param input_articles:
        List of articles
    :param use_features:
        List of features
    :return:
        Tuple of graph object eigen value and vectos
    """
    G = nx.Graph()
    features = use_features
    usage = defaultdict(int)
    for article in input_articles:

        for f in features:
            for feature in article.extracted_entities[f]:
                fs = str(feature).lower().strip()
                usage[fs] += 1
                G.add_edge(article.url, fs)
        G.add_edge(article.url, article.entity_name)

    Gc = copy(G)
    for k, v in usage.items():
        if Gc.degree(k) == 1:
            Gc.remove_node(k)
    # Create laplacian matrix and get eigen values

    L = np.asarray(nx.laplacian_matrix(Gc).todense())
    eign_val, eign_vctrs = np.linalg.eig(L)
    sorted_ind = np.argsort(eign_val)

    eign_val = eign_val[sorted_ind]
    eign_vctrs = eign_vctrs[:, sorted_ind]

    return Gc, eign_val, eign_vctrs


def get_node_eign_vector(
    g: Any,
    input_articles: List[Article],
    eign_vctrs: np.ndarray,
    vector_index_start=1,
    vector_index_end=4,
) -> pd.DataFrame():
    """
    Return dataframe of article nodes and corresponding vectors.

    :param g:
        Graph object
    :param input_articles:
        Input article list
    :param eign_vctrs:
        Eign vectors of the graph nodes
    :param vector_index_start:
    :param vector_index_end:

    :return:
        dataframe with url nodes and eigen vectors
    """
    df = nx.to_pandas_adjacency(g, dtype=int)
    vector_embedding = np.real(eign_vctrs[:, vector_index_start:vector_index_end])
    article_urls = [article.url for article in input_articles]
    df["vector"] = vector_embedding.tolist()
    df["node_val"] = df.index
    df_article = df[df.node_val.isin(article_urls)]
    # since at times the urls are repeated for an entity
    if len(article_urls) != len(df_article):
        df_repeated = get_duplicate_url_df(df_article, input_articles)
        df_article = df_article.append(df_repeated)

    df_article = df_article[["node_val", "vector"]].reset_index(drop=True)

    return df_article


def get_graph_clusters(graph_dataframe: pd.DataFrame(), th: float) -> Tuple[np.ndarray, Any, Any]:
    """
    Return similarity matrix and graph dataframe.

    :param graph_dataframe:
        The dataframe with url node eigen vectors
    :param th:
        The threshold to create graph clusters
    :return:
        Tuple of similarity matrix and graphs
    """
    article_vectors = np.array(graph_dataframe.vector.values.tolist())
    sim = metrics.pairwise.cosine_similarity(article_vectors)
    adjacency_matrix = np.zeros_like(sim)
    adjacency_matrix[sim > th] = 1

    g = nx.Graph(adjacency_matrix)
    sub_graphs = nx.connected_components(g)

    return sim, g, sub_graphs


def get_duplicate_url_df(
    df_original: pd.DataFrame, all_article_urls: List[Article]
) -> pd.DataFrame():
    """
    Return dataframe of duplicate urls and their eigen vector.

    :param df_original:
        Original url node value dataframe
    :param all_article_urls:
        list of articles
    :return:
        dataframe of duplicate urls and node vector
    """
    counter = Counter([article.url for article in all_article_urls])
    duplicate_urls = [(k, i) for k, i in counter.items() if i >= 2]
    repeated_urls = []
    for url in duplicate_urls:
        x = [url[0]] * (url[1] - 1)
        repeated_urls.append(x)

    repeated_urls = pd.DataFrame([url for urls in repeated_urls for url in urls])

    repeated_urls.columns = ["urls"]

    repeated_urls_df = pd.merge(
        repeated_urls,
        df_original[["node_val", "vector"]],
        how="inner",
        left_on="urls",
        right_on="node_val",
    )[["node_val", "vector"]]

    return repeated_urls_df
