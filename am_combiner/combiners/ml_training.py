from typing import List, Tuple, Union, Dict

import sklearn
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import json
import copy

from sklearn.preprocessing import normalize
from torch.utils.data import Dataset
from dgl.dataloading import GraphDataLoader

from am_combiner.qa.quality_metrics import validate_combiner
from am_combiner.combiners.common import (
    Combiner,
    SCORE_TO_MINIMISE_FIELD,
    BLOCKING_FIELD_FIELD,
)
from am_combiner.features.article import Article, Features
from am_combiner.features.nn.helpers import build_cluster_id_cache, binary_threshold_search
from am_combiner.features.nn.common import articles_to_homogeneous_graph
from am_combiner.combiners.ml import GCNCombiner
from am_combiner.features.nn.helpers import get_plot

TRAINING_PARAMS = {"epoch": 2, "t_min": 0.8, "t_max": 0.999, "hop_num": 3, "margin": 0.8}


def get_embeddings(G, embeddings, net, training=False):
    """Get embedding of articles based on the neural network."""
    edge_weight = GCNCombiner.get_edge_weight(G)
    if isinstance(embeddings, dict):
        embedding = {}
        for k, v in embeddings.items():
            embedding[k] = v[0]
    else:
        embedding = embeddings[0]

    if training:
        articles_emb = net(G, embedding, edge_weight)
    else:
        articles_emb = net(G, embedding, edge_weight).detach().numpy()

    return articles_emb


def get_report(sim, articles_data, input_entities_df, th):
    """Get reports of various metrics after evaluation."""
    adjacency_matrix = np.zeros_like(sim)
    adjacency_matrix[sim > th] = 1

    dataframe = Combiner.return_output_dataframe(
        cluster_ids=Combiner.compute_cluster_ids_from_adjacency_matrix(
            adjacency_matrix, articles_data, None
        ),
        unique_ids=[article_data["url"][0] for article_data in articles_data],
        blocking_names=[article_data["entity_name"][0] for article_data in articles_data],
    )
    report, clustering_quality_df = validate_combiner(
        input_entities_df[
            input_entities_df[BLOCKING_FIELD_FIELD] == articles_data[0]["entity_name"][0]
        ],
        dataframe,
        verbose=False,
    )
    return report


def calculate_score(th, net, test_data_loader, input_entities_df):
    """Combine articles produced by GCN combiner."""
    reports = []
    for G, embeddings, labels, articles_data in test_data_loader:
        article_emb = get_embeddings(G, embeddings, net)
        sim = sklearn.metrics.pairwise.cosine_similarity(normalize(article_emb, norm="l2"))
        report = get_report(sim, articles_data, input_entities_df, th)
        reports.append(report)

    df = pd.DataFrame(reports)
    df = df.mean(axis=0).to_dict()
    return df[SCORE_TO_MINIMISE_FIELD]


def get_metrics_dataframe(
    net, test_data_loader, input_entities_df, th_min: int = 50, th_max: int = 100
):
    """Return the metrics dataframe on test data for various thresholds."""
    reports = []
    for G, embeddings, labels, articles_data in test_data_loader:

        article_emb = get_embeddings(G, embeddings, net)
        sim = sklearn.metrics.pairwise.cosine_similarity(normalize(article_emb, norm="l2"))

        for th in range(th_min, th_max, 1):
            report = get_report(sim, articles_data, input_entities_df, th)
            report["threshold"] = th / 100
            reports.append(report)
    df = pd.DataFrame(reports)
    df_mean = df.groupby("threshold").mean().reset_index()

    return df_mean


def get_ground_truth_adjacency(labels: List[int]) -> torch.Tensor:
    """

    Transform labels into ground truth (n, n) adjacency matrix.

    Parameters
    ----------
    labels:
        ground truth cluster ids.

    Returns
    -------
    (n, n) matrix where (i, j) = 1 iff i and j are in the same cluster

    """
    x = np.array(labels)

    n = x.shape[0]
    ground_truth_matrix = np.zeros((n, n))
    for val in set(x):
        mask = (x == val).reshape((1, -1))
        mask_2d = np.matmul(mask.T, mask)
        ground_truth_matrix[mask_2d] = 1.0

    return torch.tensor(ground_truth_matrix, dtype=torch.float32)


def compute_pairwise_cosine_loss(
    article_representations: torch.Tensor, ground_truth_matrix: torch.Tensor, margin: float
) -> torch.Tensor:
    """

    Compute Cosine Embedding loss for entire adjacency matrix.

    Parameters
    ----------
    article_representations:
        neural network output of dim (node_num, rep_dim).
    ground_truth_matrix:
        matrix where (i, j) = 1 iff i and j are in the same cluster.
    margin
        margin below which cosine similarity is considered 'dissimilar'.

    Returns
    -------
    Sum of loss for every adjacency matrix entry.

    """
    # Compute cosine similarity in parallel:
    article_representations = F.normalize(article_representations, p=2, dim=1)
    pairwise_cosine_sim = torch.matmul(article_representations, article_representations.T)

    # Transform to 1D arrays:
    pairwise_cosine_sim = pairwise_cosine_sim.view((-1))
    ground_truth_matrix = ground_truth_matrix.view((-1))

    # Separate cosine similarity values by label
    pos_cos = pairwise_cosine_sim[ground_truth_matrix == 1]
    neg_cos = pairwise_cosine_sim[ground_truth_matrix == 0]

    # Compute torch.nn.CosineEmbeddingLoss function:
    # if label == 1, then loss is 1-cos(x1, x2)
    # if label == 0, then loss is max(0, cos(x1, x2)-margin)
    loss = (pos_cos.shape[0] - pos_cos.sum()) + F.relu(neg_cos - margin).sum()

    return loss


class GraphClustersDataset(Dataset):

    """Custom graph dataset."""

    def __init__(
        self,
        input_entities_df,
        articles_dict,
        data_agg_function,
        node_features,
        edge_features,
    ):
        """
        Store minimum data to start producing training/testing tuples.

        Parameters
        ----------
        input_entities_df:
            dataset with ground truth cluster data
        articles_dict:
            dictionary of all entity_clusters to list of articles
        data_agg_function:
            function used to create graph and convert to dense matrices
        node_features:
            parameters for data_agg_function
        edge_features:
            parameters for data_agg_function

        """
        self.input_entities_ls = list(articles_dict.values())
        self.data_agg_function = data_agg_function
        self.cid_cache = build_cluster_id_cache(input_entities_df)

        self.node_features = node_features
        self.edge_features = edge_features

    def __len__(self):
        """Get number of graphs in dataset."""
        return len(self.input_entities_ls)

    def __getitem__(self, idx):
        """
        Convert sparse matrices to dense and prepare data for NN input.

        Parameters
        ----------
        idx:
            graph index.

        Returns
        -------
        (Graph, dense node embeddings, ground truth labels, url and entity name information)

        """
        input_entities = self.input_entities_ls[idx]

        g, embedding = self.data_agg_function(
            input_entities=input_entities,
            node_feature=self.node_features,
            edge_features=self.edge_features,
        )

        labels = []
        article_data = []
        for article in input_entities:
            cluster_id = self.cid_cache[article.url]
            labels.append(cluster_id)
            article_data.append({"url": article.url, "entity_name": article.entity_name})
        ground_truth_adjacency = get_ground_truth_adjacency(labels)

        return g, embedding, ground_truth_adjacency, article_data


class GCNModelTraining:

    """Implement GCN model training."""

    def __init__(
        self,
        node_features: Union[Features, str],
        edge_features: Tuple[Union[Features, str], ...],
    ):
        self.node_features = node_features
        self.edge_features = edge_features

        self.best_net = None
        self.best_th = None
        self.best_score = float("inf")
        self.net = None
        self.df_mean_metrics = None

        self.test_data_loader = None
        self.train_data_loader = None

        self.test_df = None
        self.in_feats = None
        self.rep_dim = None

    def add_train_test_data(
        self,
        input_test_entities: pd.DataFrame,
        input_train_entities: pd.DataFrame,
        articles_dict_test: Dict[str, List[Article]],
        articles_dict_train: Dict[str, List[Article]],
        data_agg_function=articles_to_homogeneous_graph,
        num_workers: int = 0,
    ):
        """
        Use custom function to transform article dicts to tuples (G, embedding, labels, articles).

        Parameters
        ----------
        input_test_entities:
            test dataframe with ground truth Cluster ids
        input_train_entities:
            train dataframe with ground truth Cluster ids
        articles_dict_test:
            test article dictionary
        articles_dict_train:
            train article dictionary
        data_agg_function:
            function that produces (G, embedding) pair
        num_workers:
            subprocesses to use for data loading.
            If 0, then data will be loaded in the main process.

        Returns
        -------
        .train_inputs & .test_inputs inputs for graph Neural Network

        """
        test_dataset = GraphClustersDataset(
            input_test_entities,
            articles_dict_test,
            data_agg_function,
            self.node_features,
            self.edge_features,
        )
        train_dataset = GraphClustersDataset(
            input_train_entities,
            articles_dict_train,
            data_agg_function,
            self.node_features,
            self.edge_features,
        )

        self.train_data_loader = GraphDataLoader(train_dataset, num_workers=num_workers)
        self.test_data_loader = GraphDataLoader(test_dataset, num_workers=num_workers)

        self.num_classes = input_train_entities["ClusterIDGlobal"].max() + 1
        first_entity = list(articles_dict_test.keys())[0]
        self.in_feats = (
            articles_dict_test[first_entity][0].extracted_entities[self.node_features].shape[1]
        )

        self.test_df = input_test_entities

    def save_best_model(
        self,
        save_model_path: str,
        model_name: str,
    ) -> Tuple[str, str]:
        """
        Save best performing model.

        Parameters
        ----------
        save_model_path:
            path to the folder where to save model
        model_name
            name of the model
        Returns
        -------

        """
        full_model_name = f"{model_name}_th_{round(self.best_th, 2)}.torch"
        full_json_name = f"{model_name}_th_{round(self.best_th, 2)}.json"
        model_path = os.path.join(save_model_path, full_model_name)
        torch.save(self.best_net.state_dict(), model_path)
        config_path = os.path.join(save_model_path, full_json_name)
        with open(config_path, "w+") as f:
            json.dump(
                {
                    "best_th": self.best_th,
                    "in_feats": self.in_feats,
                    "rep_dim": self.rep_dim,
                },
                f,
            )
        return model_path, config_path

    def train_model(
        self,
        training_params=TRAINING_PARAMS,
        auc_check=False,
        verbose=False,
        th_min: int = 50,
        th_max: int = 100,
    ) -> float:
        """
        Train model.

        Parameters
        ----------
        training_params
            dictionary of training parameters.
        auc_check
            check output for various thresholds for each epoch and get graphs and auc score.
        verbose
            display test set results as the training progresses.
        th_min:
            minimum threshold.
        th_max:
            maximum threshold.

        Returns
        -------
            best score on test dataset.

        """
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        margin = training_params["margin"]
        self.df_mean_metrics = pd.DataFrame()
        for epoch in range(training_params["epoch"]):
            for G, embeddings, ground_truth_adjacency, _ in self.train_data_loader:
                article_representations = get_embeddings(G, embeddings, self.net, training=True)

                loss = compute_pairwise_cosine_loss(
                    article_representations, ground_truth_adjacency[0], margin
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Get the metrics for threshold range
            if auc_check:
                df_metrics = get_metrics_dataframe(
                    self.net, self.test_data_loader, self.test_df, th_min, th_max
                )
                df_metrics["epoch"] = epoch
                self.df_mean_metrics = self.df_mean_metrics.append(df_metrics)

            # Binary Search for best threshold
            tried_x, tried_y = binary_threshold_search(
                self.net,
                self.test_data_loader,
                self.test_df,
                calculate_score,
                t_min=training_params["t_min"],
                t_max=training_params["t_max"],
                hop_num=training_params["hop_num"],
            )

            if verbose:
                print(f"Epoch {epoch}: thresholds: {tried_x} gives scores: {tried_y}")

            if min(tried_y) < self.best_score:
                self.best_score = min(tried_y)
                self.best_net = copy.deepcopy(self.net)
                self.best_th = tried_x[np.argmin(tried_y)]

        self.rep_dim = article_representations.shape[1]

        if auc_check:
            print(get_plot("Homogeneity", "Completeness", self.df_mean_metrics))
            print(get_plot("Name UC rate", "Name OC rate", self.df_mean_metrics))

        return self.best_score
