from collections import defaultdict
from typing import List, Iterable, Union, Tuple, Optional
from scipy.sparse import coo_matrix

import itertools
import numpy as np

from am_combiner.features.article import Article, Features
from am_combiner.features.sanction import Sanction, SanctionFeatures

CombinedObject = Union[Article, Sanction]
CombinedObjectFeature = Union[Features, SanctionFeatures]


def get_article_feature_adjacency_matrix(
    articles: List[Article],
    feature: Features,
    inverse_degree: bool = False,
) -> np.array:
    """
    Build an adjacency matrix for a set of articles for a given features.

    This is an O(N*k + E) procedure that loops through articles and looks at features intersections,
    where
        N is a number of articles,
        k is average number of entities per article
        E is number of edges.
    The adjacency matrix will also contain the strength of connections,
    i.e. how many common features each articles pair had.

    Parameters
    ----------
    articles: List[Articles]
        A list of articles that adjacency matrix will be built for
    feature: Features
        which feature to use to built the matrix.
    inverse_degree: bool
        If true, add weight to adjacency matrix, where edge is inversely proportionate
        to the frequency of feature value.

    Returns
    -------
    np.array
        representing the adjacency matrix

    """
    row, col, data = [], [], []
    num_entries = len(articles)

    extracted_values_to_article = defaultdict(list)

    for i, article in enumerate(articles):
        for extracted_value in article.extracted_entities[feature]:
            extracted_values_to_article[extracted_value].append(i)

    for article_id_ls in extracted_values_to_article.values():
        for i, j in itertools.permutations(set(article_id_ls), 2):
            row.append(i)
            col.append(j)

            if inverse_degree:
                value = 1.0 / len(article_id_ls)
            else:
                value = 1.0
            data.append(value)

    adjacency = coo_matrix((data, (row, col)), shape=(num_entries, num_entries))
    adjacency.sum_duplicates()
    adjacency.eliminate_zeros()
    return adjacency


def get_article_multi_feature_adjacency(
    articles: List[CombinedObject],
    features: Iterable[CombinedObjectFeature],
    as_list: bool = False,
    inverse_degree: bool = False,
) -> np.array:
    """
    Build an adjacency matrix for a set of articles for a given list of features.

    This is an N^2 procedure that loops through articles and looks at features intersections.
    The adjacency matrix will also contain the strength of connections,
    i.e. how many common features each articles pair had.

    Parameters
    ----------
    articles: List[Articles]
        A list of articles that adjacency matrix will be built for
    features: List[Features]
        List of features for which to built the matrix.
    as_list: bool
        if True, the result is a list of adjacency matrices for each feature, otherwise cumulative
        matrix is returned.
    inverse_degree: bool
        If true, add weight to adjacency matrix, where edge is inversely proportionate
        to the frequency of feature value.

    Returns
    -------
    np.array
        representing the adjacency matrix

    """
    all_adjacent = []
    for f in features:
        this_adjacency = get_article_feature_adjacency_matrix(articles, f, inverse_degree)
        all_adjacent.append(this_adjacency)
    if as_list:
        return all_adjacent
    else:
        adjacent_sum = np.array(all_adjacent).sum(axis=0)
        adjacent_sum = coo_matrix(adjacent_sum)
    return adjacent_sum


def get_feature_negative_edge_matrix(
    objects: List[CombinedObject],
    feature: CombinedObjectFeature,
    distance: Optional[int],
) -> np.array:
    """
    Build negative adjacency matrix for a set of articles for a given features.

    Parameters
    ----------
    objects:
        A list of articles that adjacency matrix will be built for
    feature:
        which feature to use to built the matrix.
    distance:
        If None, require exact match. Else, distance should be greater than.

    Returns
    -------
    np.array
        representing the adjacency matrix

    """
    num_entries = len(objects)
    non_empty_indices = []
    negative_adjacency = np.zeros((num_entries, num_entries))

    for inx, object in enumerate(objects):
        if object.extracted_entities[feature]:
            non_empty_indices.append(inx)

    for i in non_empty_indices:
        for j in non_empty_indices:

            if i == j:
                continue

            set1 = objects[i].extracted_entities[feature]
            set2 = objects[j].extracted_entities[feature]

            add_negative = False
            if distance is None:
                if not set1.intersection(set2):
                    add_negative = True
            else:
                # WLOG min(set1) <= min(set2)
                if min(set1) > min(set2):
                    set1, set2 = set2, set1

                if min(set2) - max(set1) > distance:
                    add_negative = True

            if add_negative:
                negative_adjacency[j, i] = 1

    return negative_adjacency


def get_multi_feature_negative_edges(
    entities: List[CombinedObject],
    negators: List[Tuple[CombinedObjectFeature, Optional[int]]],
) -> np.array:
    """
    Build negative adjacency matrix for a set of articles for a given list of features.

    Parameters
    ----------
    entities:
        A list of articles or sanctions that adjacency matrix will be built for
    negators:
        Tuples of (feature, distance), where
        distance: None means that lack of distinct match implies negative edge
        distance: int create negative edge if distance is exceeded.

    Returns
    -------
    np.array
        representing the adjacency matrix

    """
    all_adjacent = []
    for feature, distance in negators:
        all_adjacent.append(get_feature_negative_edge_matrix(entities, feature, distance))
    return np.array(all_adjacent).sum(axis=0)
