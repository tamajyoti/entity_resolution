import numpy as np
import pandas as pd
import math
import torch
import pytest

from scipy import sparse
from torch import tensor
from dgl.dataloading import GraphDataLoader

from am_combiner.combiners.ml import GCN, HeteroGCN
from am_combiner.features.article import Features, Article
from am_combiner.combiners.ml_training import (
    calculate_score,
    GCNModelTraining,
    GraphClustersDataset,
    get_ground_truth_adjacency,
    compute_pairwise_cosine_loss,
)
from am_combiner.features.nn.common import (
    articles_to_homogeneous_graph,
    articles_to_hetero_graph,
    HETEROGENEOUS_NODE_NAME,
)
from am_combiner.combiners.common import (
    UNIQUE_ID_FIELD,
    BLOCKING_FIELD_FIELD,
    GROUND_TRUTH_FIELD,
    CLUSTER_ID_GLOBAL_FIELD,
)


def gen_sparse_matrix(shape):
    return sparse.csr_matrix(np.random.random(shape).clip(0.8) - 0.8)


def get_fake_data_articles(n):

    # Initiate Articles
    articles = []
    urls = []
    for i in range(n):
        articles.append(Article("A", "", f"url{i}"))
        urls.append(f"url{i}")

    np.random.seed(0)
    for i, article in enumerate(articles):
        article.extracted_entities[Features.TFIDF_FULL_TEXT_12000] = gen_sparse_matrix(12000)

        # Add edge features:
        for f in [Features.ORG_CLEAN, Features.PERSON_CLEAN]:
            article.extracted_entities[f] = np.random.choice(range(10), 3)

    input_df = pd.DataFrame(
        data={
            UNIQUE_ID_FIELD: urls,
            BLOCKING_FIELD_FIELD: ["A"] * n,
            GROUND_TRUTH_FIELD: list(range(n)),
            CLUSTER_ID_GLOBAL_FIELD: list(range(n)),
        }
    )
    return {"A": articles}, input_df


def test_add_data():
    training = GCNModelTraining(
        node_features=Features.TFIDF_FULL_TEXT_12000,
        edge_features=[Features.ORG_CLEAN, Features.PERSON_CLEAN],
    )
    test_articles, test_input_df = get_fake_data_articles(3)
    train_articles, train_input_df = get_fake_data_articles(5)

    training.add_train_test_data(
        test_input_df,
        train_input_df,
        test_articles,
        train_articles,
        articles_to_homogeneous_graph,
    )

    assert training.in_feats == 12000
    assert training.num_classes == 5


def test_graph_dataset():

    articles_dict, input_entities_df = get_fake_data_articles(3)

    gcd = GraphClustersDataset(
        input_entities_df,
        articles_dict,
        articles_to_homogeneous_graph,
        Features.TFIDF_FULL_TEXT_12000,
        [Features.ORG_CLEAN, Features.PERSON_CLEAN],
    )

    assert gcd.__len__() == 1

    g, embedding, ground_truth_adj, article_data = gcd.__getitem__(0)
    assert g.num_nodes() == 3
    assert list(embedding.shape) == [3, 12000]
    assert ground_truth_adj.shape == (3, 3)
    assert len(article_data) == 3


@pytest.mark.parametrize(
    "input_label, expected_output_matrix",
    [
        ([17, 17], np.array([[1, 1], [1, 1]])),
        ([0, 1], np.array([[1, 0], [0, 1]])),
        ([3, 3, 1], np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])),
        ([0, 1, 2, 1], np.array([[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 1, 0, 1]])),
    ],
)
def test_get_ground_truth_adjacency(input_label, expected_output_matrix):
    output_matrix = get_ground_truth_adjacency(input_label).detach().numpy()
    assert (output_matrix == expected_output_matrix).all()


@pytest.mark.parametrize(
    "expected_result, margin, ground_truth, article_rep",
    [
        (0, 0, torch.Tensor([[1, 0], [0, 1]]), torch.Tensor([[1, 0], [0, 1]])),
        (2, -1, torch.Tensor([[1, 0], [0, 1]]), torch.Tensor([[1, 0], [0, 1]])),
        (2, 0, torch.Tensor([[1, 1], [1, 1]]), torch.Tensor([[1, 0], [0, 1]])),
        (1.4, 0.3, torch.Tensor([[1, 0], [0, 1]]), torch.Tensor([[1, 1], [2, 2]])),
        (0, 0.3, torch.Tensor([[1, 1], [1, 1]]), torch.Tensor([[1, 1], [2, 2]])),
        (4, 0, torch.Tensor([[1, 1], [1, 1]]), torch.Tensor([[-1, -1], [2, 2]])),
    ],
)
def test_compute_pairwise_cosine_loss(expected_result, margin, ground_truth, article_rep):
    true_result = compute_pairwise_cosine_loss(article_rep, ground_truth, margin)
    assert expected_result == round(true_result.item(), 2)


def test_graph_dataset_hetero():

    articles_dict, input_entities_df = get_fake_data_articles(3)

    gcd = GraphClustersDataset(
        input_entities_df,
        articles_dict,
        articles_to_hetero_graph,
        Features.TFIDF_FULL_TEXT_12000,
        [Features.ORG_CLEAN, Features.PERSON_CLEAN],
    )

    assert gcd.__len__() == 1

    g, embedding, ground_truth_adjacency, article_data = gcd.__getitem__(0)
    assert g.num_nodes() == 3
    assert list(embedding[HETEROGENEOUS_NODE_NAME].shape) == [3, 12000]
    assert list(ground_truth_adjacency.shape) == [3, 3]
    assert len(article_data) == 3


def test_training_homogeneous_graph():

    training = GCNModelTraining(
        node_features=Features.TFIDF_FULL_TEXT_12000,
        edge_features=[Features.ORG_CLEAN, Features.PERSON_CLEAN],
    )

    assert math.isinf(training.best_score)

    test_articles, test_input_df = get_fake_data_articles(3)
    train_articles, train_input_df = get_fake_data_articles(5)

    training.add_train_test_data(
        test_input_df,
        train_input_df,
        test_articles,
        train_articles,
        articles_to_homogeneous_graph,
    )
    training.net = GCN(12000, 10)
    training.train_model(
        training_params={"epoch": 1, "t_min": 0.8, "t_max": 0.9, "hop_num": 1, "margin": 0}
    )

    assert training.best_th in [0.8, 0.85, 0.9]
    assert not math.isinf(training.best_score)


def test_training_heterogeneous_graph():
    training = GCNModelTraining(
        node_features=Features.TFIDF_FULL_TEXT_12000,
        edge_features=[Features.ORG_CLEAN, Features.PERSON_CLEAN],
    )

    assert math.isinf(training.best_score)

    test_articles, test_input_df = get_fake_data_articles(3)
    train_articles, train_input_df = get_fake_data_articles(5)

    training.add_train_test_data(
        test_input_df,
        train_input_df,
        test_articles,
        train_articles,
        articles_to_hetero_graph,
    )
    training.net = HeteroGCN(["ORG_CLEAN", "PERSON_CLEAN"], 12000, 10)
    training.train_model(
        training_params={"epoch": 1, "t_min": 0.8, "t_max": 0.9, "hop_num": 1, "margin": 0}
    )

    assert training.best_th in [0.8, 0.85, 0.9]
    assert not math.isinf(training.best_score)


def test_calculate_score_homogenous_all_same_cluster():
    test_articles, test_input_df = get_fake_data_articles(3)
    test_input_df[GROUND_TRUTH_FIELD] = [0, 0, 0]

    gcd = GraphClustersDataset(
        test_input_df,
        test_articles,
        articles_to_homogeneous_graph,
        Features.TFIDF_FULL_TEXT_12000,
        [Features.ORG_CLEAN],
    )

    test_data_loader = GraphDataLoader(gcd)

    def net_func(_g, _embedding, _edge_weight):
        return tensor(np.ones((3, 10)))

    score = calculate_score(0.9, net_func, test_data_loader, test_input_df)

    assert score == 0.0


def test_calculate_score_hetero_all_different_clusters():
    test_articles, test_input_df = get_fake_data_articles(3)

    gcd = GraphClustersDataset(
        test_input_df,
        test_articles,
        articles_to_hetero_graph,
        Features.TFIDF_FULL_TEXT_12000,
        [Features.ORG_CLEAN],
    )

    test_data_loader = GraphDataLoader(gcd)

    def net_func(_g, _embedding, _edge_weight):
        return tensor(np.random.rand(3, 1000))

    score = calculate_score(0.9, net_func, test_data_loader, test_input_df)

    assert score == 0.0
