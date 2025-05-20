import pickle
from typing import List, Union, Optional

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import json
import dgl

from dgl.nn.pytorch.conv import SAGEConv
from pymongo import MongoClient
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack as sparse_vstack
from am_combiner.combiners.common import Combiner
from am_combiner.features.article import Article, Features
from am_combiner.features.nn.common import (
    articles_to_homogeneous_graph,
    articles_to_hetero_graph,
    HETEROGENEOUS_NODE_NAME,
)
from am_combiner.utils.storage import ensure_s3_resource_exists, store_similarities
from am_combiner.splitters.common import Splitter

LARGE_CLUSTER_TH_BUMP_UP = 0.02
LARGE_CLUSTER_LIMIT = 400


class GCN(torch.nn.Module):

    """Simple graph convolution network implementation."""

    def __init__(self, in_feats, rep_dim):
        super(GCN, self).__init__()
        self.conv1 = SAGEConv(
            in_feats, rep_dim, aggregator_type="mean", activation=torch.nn.LeakyReLU()
        )

    def forward(self, g, node_embeddings, edge_weight):
        """Implement forward propagation of the torch module object."""
        return self.conv1(g, node_embeddings, edge_weight)


class HeteroGCN(torch.nn.Module):

    """Heterogeneous graph convolution network implementation."""

    def __init__(self, mods, in_feats, rep_dim):
        super(HeteroGCN, self).__init__()
        conv_dict = {}
        for mod in mods:
            conv_dict[mod] = SAGEConv(
                in_feats, rep_dim, aggregator_type="mean", activation=torch.nn.ReLU()
            )
        self.conv = dgl.nn.HeteroGraphConv(mods=conv_dict, aggregate="mean")

    def forward(self, graph, embeddings, edge_weights):
        """Implement forward propagation of the torch module object."""
        embeddings = self.conv(graph, embeddings, mod_kwargs=edge_weights)
        return embeddings[HETEROGENEOUS_NODE_NAME]


class GCNCombiner(Combiner):

    """Combines entities with sage convs."""

    def __init__(
        self,
        use_features: List[Features],
        node_features: Union[Features, str],
        model_uri: str,
        config_uri: str,
        cache: str,
        mongo_uri: Optional[str] = None,
        mongo_collection: Optional[str] = None,
    ):
        super().__init__()
        self.use_features = use_features
        self.node_features = node_features
        model_path = ensure_s3_resource_exists(model_uri, cache)
        self._model_path = model_path
        config_path = ensure_s3_resource_exists(config_uri, cache)
        self._config_path = config_path
        self.th = None
        self._load_model()

        self.mongo_client = None
        if mongo_uri and mongo_collection:
            self.mongo_client = MongoClient(mongo_uri).get_database()[mongo_collection]

    def _load_model(self):
        """Load the model with the given config."""
        with open(self._config_path, "r") as f:
            config = json.load(f)
            self.th = config["best_th"]
            rep_dim = config["rep_dim"]
            in_feats = config["in_feats"]

        self.net = GCN(in_feats, rep_dim)
        self.net.load_state_dict(torch.load(self._model_path))
        self.net.eval()

    def _get_graph_and_features(self, input_entities: List[Article]):
        """Get features and graph representation."""
        return articles_to_homogeneous_graph(
            input_entities=input_entities,
            node_feature=self.node_features,
            edge_features=self.use_features,
        )

    @staticmethod
    def get_edge_weight(g):
        """
        Retrieve edge_weight parameter to feed additional data to GCN.

        Parameters
        ----------
        g:
            graph (either homogeneous or heterogeneous, with or without edge weights).

        Returns
        -------
            edge_weight tensor if edge weights exists (and None otherwise).

        """
        if g.is_homogeneous:
            return g.edata["weight"].float()

        edge_weights = {}
        for edge_type in g.etypes:
            edge_weights[edge_type] = {"edge_weight": g[edge_type].edata["weight"]}
        return edge_weights

    def _enhance_pairwise_similarities(
        self, sim: np.ndarray, input_entities: List[Article]
    ) -> np.ndarray:
        """
        Enhance the initial pairwise similarities of the input entities.

        This proxy function is needed since in derived classes one may want to change the logic of
        how the pairwise similarities are built and therefore use additional features from articles
        to boost/ reduce the similarity score.

        Parameters
        ----------
        sim:
            A matrix representing initial pairwise similarities.
        input_entities:
            The input entities to enhance pairwise similarities for.

        Returns
        -------
            Enhanced pairwise similarities.

        """
        return sim

    def _get_adjacency_from_similarities(
        self, sim: np.ndarray, splitter: Optional[Splitter] = None
    ) -> np.ndarray:
        """
        Use pairwise similarities to build an adjacency matrix according to the threshold.

        Parameters
        ----------
        sim:
            A matrix representing pairwise similarities.
        splitter:
            Splitter on resulting adjacency matrix.

        Returns
        -------
            An array representing the adjacency matrix.

        """
        # This is special threshold add on for extra large clusters.
        # Otherwise, large clusters get over-combined due to large absolute
        # number of false positive connections that scale with article number.
        if sim.shape[0] > LARGE_CLUSTER_LIMIT:
            th_addon = LARGE_CLUSTER_TH_BUMP_UP
        else:
            th_addon = 0
        adjacency_matrix = np.zeros_like(sim)
        adjacency_matrix[sim > self.th + th_addon] = 1
        return adjacency_matrix

    def combine_entities(
        self, input_entities: List[Article], splitter: Optional[Splitter] = None
    ) -> pd.DataFrame:
        """Concrete implementation of the abstract method."""
        G, embeddings = self._get_graph_and_features(input_entities=input_entities)
        article_emb = self.net(G, embeddings, self.get_edge_weight(G)).detach().numpy()
        sim = sklearn.metrics.pairwise.cosine_similarity(normalize(article_emb, norm="l2"))
        sim = self._enhance_pairwise_similarities(sim, input_entities)
        adjacency_matrix = self._get_adjacency_from_similarities(sim)
        cluster_ids = Combiner.compute_cluster_ids_from_adjacency_matrix(
            adjacency_matrix, input_entities, splitter
        )

        if self.mongo_client:
            store_similarities(sim, input_entities, cluster_ids, self.mongo_client)

        return Combiner.return_output_dataframe(
            cluster_ids=cluster_ids,
            unique_ids=[article.url for article in input_entities],
            blocking_names=[article.entity_name for article in input_entities],
        )


class GCNHeteroCombiner(GCNCombiner):

    """Use heterogeneous graph convs to create article representations."""

    def _load_model(self):
        mods = [str(edge_feature).split(".")[-1] for edge_feature in self.use_features]
        with open(self._config_path, "r") as f:
            config = json.load(f)
            self.th = config["best_th"]
            rep_dim = config["rep_dim"]
            in_feats = config["in_feats"]

        self.net = HeteroGCN(mods, in_feats, rep_dim)
        self.net.load_state_dict(torch.load(self._model_path))
        self.net.eval()

    def _get_graph_and_features(self, input_entities: List[Article]):
        """Get features and graph representations."""
        return articles_to_hetero_graph(
            input_entities=input_entities,
            node_feature=self.node_features,
            edge_features=self.use_features,
        )


class GCNCombinerWithLinearCombination(GCNHeteroCombiner):

    """Use heterogeneous graph convs and linear combinations to create article representations."""

    def __init__(
        self,
        use_features: List[Features],
        node_features: Union[Features, str],
        th: float,
        model_uri: str,
        config_uri: str,
        lc_uri: str,
        cache: str,
    ):
        super().__init__(use_features, node_features, model_uri, config_uri, cache)
        lc_path = ensure_s3_resource_exists(lc_uri, cache)
        self._lc_path = lc_path
        with open(lc_path, "rb") as f:
            self.lc = pickle.load(f)
        # Note, this th overrides the th read from config json.
        # This is by design, the meaning of LC th is different.
        self.th = th

    def _enhance_pairwise_similarities(
        self, sim: np.ndarray, input_entities: List[Article]
    ) -> np.ndarray:
        tfidf_embs = sparse_vstack(
            [ie.extracted_entities[self.node_features] for ie in input_entities]
        )
        tfidf_sim = cosine_similarity(normalize(tfidf_embs, norm="l2"))

        mini_features = np.hstack((tfidf_sim.reshape(-1, 1), sim.reshape(-1, 1)))
        num = len(input_entities)
        enh_sim = self.lc.predict_proba(mini_features)[:, 1].reshape(num, num)
        return enh_sim


class SklearnClassificationModelBasedCombiner(Combiner):

    """
    A concrete implementation of the Combiner abstract class.

    Implements a combiner that combines entities using a classification model from sklearn library.

    """

    def __init__(self, svm_path: str, th: float = 0.9):
        super().__init__()
        self.th = th
        with open(svm_path, "rb") as f:
            self.clf = pickle.load(f)

    def combine_entities(
        self, input_entities: List[Article], splitter: Optional[Splitter] = None
    ) -> pd.DataFrame:
        """
        Combine a list of given articles into clusters.

        Parameters
        ----------
        input_entities:
            A list of Article objects to be combined.
        splitter:
            Splitter on resulting adjacency matrix.

        Returns
        -------
            A pd.DataFrame object with cluster ids assigned to all articles.

        """

        def s2d(v: scipy.sparse.csr.csr_matrix) -> np.array:
            """
            Transform a sparse scipy matrix into a dense numpy array.

            Parameters
            ----------
            v:
                The input matrix.

            Returns
            -------
                Dense numpy array obtained by squeezing the input matrix.

            """
            return np.asarray(v.todense()).squeeze()

        adjacency_matrix = np.zeros(shape=(len(input_entities), len(input_entities)))
        for ct1, a1 in enumerate(input_entities):
            for ct2, a2 in enumerate(input_entities):
                if a1 is a2:
                    adjacency_matrix[ct1][ct2] = 1
                    continue
                v1 = s2d(a1.extracted_entities[Features.TFIDF_FULL_TEXT])
                v2 = s2d(a2.extracted_entities[Features.TFIDF_FULL_TEXT])

                prob_same = self.clf.predict_proba([np.hstack((v1, v2))])[0][1]

                connect = int(prob_same > self.th)
                adjacency_matrix[ct1][ct2] = max(connect, adjacency_matrix[ct1][ct2])
                adjacency_matrix[ct2][ct1] = max(connect, adjacency_matrix[ct2][ct1])

        return Combiner.return_output_dataframe(
            cluster_ids=Combiner.compute_cluster_ids_from_adjacency_matrix(
                adjacency_matrix, input_entities, splitter
            ),
            unique_ids=[article.url for article in input_entities],
            blocking_names=[article.entity_name for article in input_entities],
        )
