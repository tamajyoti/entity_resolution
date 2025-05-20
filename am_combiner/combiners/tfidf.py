from typing import Any, Callable, Dict, List, Union, Optional

import networkx as nx
import numpy as np
import pandas as pd
from pymongo import MongoClient
from scipy.sparse import vstack as sparse_vstack
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from am_combiner.combiners.common import Combiner
from am_combiner.utils.adjacency import get_article_feature_adjacency_matrix
from am_combiner.features.article import Article, Features
from am_combiner.utils import spectral
from am_combiner.utils.storage import store_similarities
from am_combiner.splitters.common import Splitter


class TFIDFKMeansCombiner(Combiner):

    """
    A concrete implementation of an abstract class.

    Implements a combiner that combines entities using tfidf representation and clustering.
    """

    def __init__(self, source_feature: Union[Features, str] = Features.TFIDF_FULL_TEXT):
        self.source_feature = source_feature

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
            Unused splitter due to lack of adjacency matrix.

        Returns
        -------
            A pd.DataFrame object with cluster ids assigned to all articles.

        """
        # the silhouette approach can't solve small clusters
        # so we just naively don't combine the names
        if len(input_entities) <= 3:
            clusters = [0, 1, 2][: len(input_entities)]
        else:
            vectors = sparse_vstack(
                [ie.extracted_entities[self.source_feature] for ie in input_entities]
            )
            max_coeff = 0

            for n in range(2, len(input_entities)):
                labels = (
                    MiniBatchKMeans(n_clusters=n, init_size=1024, batch_size=2048, random_state=20)
                    .fit_predict(vectors)
                    .tolist()
                )
                sil_coeff = silhouette_score(vectors, labels, metric="euclidean")

                if sil_coeff > max_coeff:
                    max_coeff = sil_coeff
                    clusters = labels

            if max_coeff == 0:
                clusters = labels

        return Combiner.return_output_dataframe(
            cluster_ids=clusters,
            unique_ids=[article.url for article in input_entities],
            blocking_names=[article.entity_name for article in input_entities],
        )


def identity_tokenizer(text: str) -> List[str]:
    """
    Split a text by _-_ separator.

    Parameters
    ----------
    text:
        The text to be tokenized.

    Returns
    -------
        Tokenized text.

    """
    return text.split("_-_")


def get_features_from_article(article: Article) -> str:
    """
    Extract the features from an article in a text form.

    Parameters
    ----------
    article
        The article.

    Returns
    -------
        A string with the features.

    """
    article_features = list()
    for feature_name, value_list in article.extracted_entities.items():
        for v in value_list:
            article_features.append(feature_name.name + "-" + str(v))
    return "_-_".join(article_features)


class TFIDFCombinerWithClusteringAlgo(Combiner):

    """
    A concrete implementation of an abstract class.

    Implements a combiner that combines entities using any clustering algo.

    """

    def __init__(self, algorithm: Callable, args: Any, kwargs: Dict):
        super().__init__()
        self.algorithm = algorithm
        self.args = args
        self.kwargs = kwargs

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
            Unused splitter due to lack of adjacency matrix.

        Returns
        -------
            A pd.DataFrame object with cluster ids assigned to all articles.

        """
        vectors = sparse_vstack(
            [ie.extracted_entities[Features.TFIDF_FULL_TEXT] for ie in input_entities]
        )

        # clusters are obtained using affinity propagation with values provided as inputs
        clusters = self.algorithm(*self.args, **self.kwargs).fit_predict(vectors.toarray())

        # Basically everything with the same name goes into the same cluster
        return Combiner.return_output_dataframe(
            cluster_ids=clusters,
            unique_ids=[article.url for article in input_entities],
            blocking_names=[article.entity_name for article in input_entities],
        )


class TFIDFCosineSimilarityCombiner(Combiner):

    """
    A concrete implementation of a combiner class.

    Does clustering based on the connected components approach and utilises cosine similarity
    for creating graph connections.

    Attributes
    ----------
    th: float
        Cosine similarity threshold.
    source_feature: Union[Features, str]
        Source feature, by default TFIDF.

    """

    def __init__(
        self,
        th: float = 0.5,
        source_feature: Union[Features, str] = Features.TFIDF_FULL_TEXT,
        mongo_uri: Optional[str] = None,
        mongo_collection: Optional[str] = None,
    ):
        super().__init__()
        self.th = th
        self.source_feature = source_feature

        self.mongo_client = None
        if mongo_uri and mongo_collection:
            self.mongo_client = MongoClient(mongo_uri).get_database()[mongo_collection]

    def _get_pairwise_similarities(self, input_entities: List[Article]) -> np.ndarray:
        """
        Get the pairwise cosine similarities of the input entities.

        Parameters
        ----------
        input_entities:
            The input entities to compute pairwise cosine similarities for.

        Returns
        -------
            The pairwise cosine similarities.

        """
        vectors = sparse_vstack(
            [ie.extracted_entities[self.source_feature] for ie in input_entities]
        )

        return cosine_similarity(vectors)

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

    def _get_adjacency_from_similarities(self, sim: np.ndarray) -> np.ndarray:
        """
        Use pairwise similarities to build an adjacency matrix according to the threshold.

        Parameters
        ----------
        sim:
            A matrix representing pairwise similarities.

        Returns
        -------
            An array representing the adjacency matrix.

        """
        adjacency_matrix = np.zeros_like(sim)
        adjacency_matrix[sim > self.th] = 1
        return adjacency_matrix

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
        sim = self._get_pairwise_similarities(input_entities)
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


class TFIDFAndFeaturesCosineSimilarityCombiner(TFIDFCosineSimilarityCombiner):

    """
    A concrete implementation of a combiner class.

    Does clustering based on feature-wise cosine similarity.

    """

    def __init__(
        self,
        use_features: List[Features],
        th: float = 0.5,
        max_energy: int = 75,
        source_feature: Union[Features, str] = Features.TFIDF_FULL_TEXT,
        mongo_uri: Optional[str] = None,
        mongo_collection: Optional[str] = None,
    ):
        super().__init__(th, source_feature, mongo_uri, mongo_collection)
        self.max_energy = max_energy
        self.use_features = use_features

    def _enhance_pairwise_similarities(
        self, sim: np.ndarray, input_entities: List[Article]
    ) -> np.ndarray:
        """
        Enhance the initial pairwise similarities of the input entities.

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
        num_entries = len(input_entities)
        all_features = self.use_features
        # This is the matrix that is used for similarity score enhancement(i.e. boosting)
        similarity_enh = np.zeros(shape=(num_entries, num_entries), dtype=np.int32)
        # We go through all features and calculate their intersection sizes
        # (i.e. how many feature values are common)
        for f in all_features:
            this_adjacency = get_article_feature_adjacency_matrix(input_entities, f)
            # Accumulating all boosting from all features into a single matrix
            similarity_enh += this_adjacency
        # Since there might be a lot of features in common, we certainly want to limit the ability
        # of this matrix to boost the cosine similarities too much,
        # hence we limit the maximum impact it can make.
        similarity_enh = np.clip(similarity_enh, 0, self.max_energy)
        # Now we scale the the boosting value so that it does not exceed
        # the maximum allowed boosting values
        sim += similarity_enh / self.max_energy
        return sim


class TFIDFAndGraphCosineSimilarityCombiner(TFIDFCosineSimilarityCombiner):

    """
    A concrete implementation of a combiner class.

    The combiner class accepts a list of features and created a connected component graph
    by connecting all nodes to a single pseudo entity vector and calculate eigen decomposition
    of the spectral graph. The eigen vector of the nodes are then used to calculate cosine
    similarity between them and then is used to enhance the overall tfidf cosine similarity
    matrix of the user.

    Does clustering based on feature-wise cosine similarity.

    use_features: The list of extracted features used in the combiner
    th: The threshold to determine if two mentions' are similar as per the similarity score
    max_energy: the maximum the spectral eigen vector can contribute to the tfidf similarity score
    min_energy: the minimum support of the eigen vector for the similarity score
    source_feature: the feature on which the tfidf cosine similarity is calculated
    """

    def __init__(
        self,
        use_features: List[Features],
        th: float = 0.5,
        max_energy: float = 0.65,
        min_energy: float = -0.25,
        source_feature: Union[Features, str] = Features.TFIDF_FULL_TEXT,
        mongo_uri: Optional[str] = None,
        mongo_collection: Optional[str] = None,
    ):
        super().__init__(th, source_feature, mongo_uri, mongo_collection)
        self.max_energy = max_energy
        self.min_energy = min_energy
        self.use_features = use_features

    def _enhance_pairwise_similarities(
        self, sim: np.ndarray, input_entities: List[Article]
    ) -> np.ndarray:
        """
        Enhance the initial pairwise similarities of the input entities.

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
        # get the graph eign vectors
        overall_graph, eign_val, eign_vctrs = spectral.get_graph_eign(
            input_entities, self.use_features
        )
        # get the graph dataframe of the nodes
        graph_df = spectral.get_node_eign_vector(overall_graph, input_entities, eign_vctrs, 1, 4)
        # get graph similarity matrix
        similarity_enh, final_graph, sub_graphs = spectral.get_graph_clusters(graph_df, self.th)
        similarity_enh = np.clip(similarity_enh, self.min_energy, self.max_energy)
        # Now we scale the the boosting value so that it does not exceed
        # the maximum allowed boosting values
        sim += similarity_enh
        return sim


class TFIDFFeatrGraphCosineSimilarityCombiner(TFIDFCosineSimilarityCombiner):

    """
    A concrete implementation of a combiner class.

    The combiner class accepts a list of features and created a connected component graph
    If the no of nodes are more than a threshold then those connected component clusters'
    spectral graph eigen decomposition is done. The eigen vector of the nodes are then used to
    calculate cosine similarity between them and then is used to enhance the overall
    tfidf cosine similarity matrix of the user.

    Does clustering based on feature-wise cosine similarity.

    use_features: The list of extracted features used in the combiner
    th: The threshold to determine if two mentions' are similar as per the similarity score
    ftr_th: The minimum number of connected components to be considered for connection
    ftr_max_energy: the maximum value for rationalizing the connected components
    graph_node: the minimum no of mentions for which spectral eigen decomposition will be done
    max_energy: the maximum the spectral eigen vector can contribute to the tfidf similarity score
    min_energy: the minimum support of the eigen vector for the similarity score
    source_feature: the feature on which the tfidf cosine similarity is calculated

    """

    def __init__(
        self,
        use_features: List[Features],
        th: float = 0.4,
        ftr_th=1,
        ftr_max_energy=75,
        graph_node_th=5,
        max_energy: float = 0.15,
        min_energy: float = 0,
        source_feature: Union[Features, str] = Features.TFIDF_FULL_TEXT,
        mongo_uri: Optional[str] = None,
        mongo_collection: Optional[str] = None,
    ):
        super().__init__(th, source_feature, mongo_uri, mongo_collection)
        self.ftr_th = ftr_th
        self.ftr_max_energy = ftr_max_energy
        self.graph_node_th = graph_node_th
        self.max_energy = max_energy
        self.min_energy = min_energy
        self.use_features = use_features

    def _enhance_pairwise_similarities(
        self, sim: np.ndarray, input_entities: List[Article]
    ) -> np.ndarray:
        """
        Enhance the initial pairwise similarities of the input entities.

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
        num_entries = len(input_entities)
        similarity_enh = np.zeros(shape=(num_entries, num_entries), dtype=np.int32)
        # We go through all features and calculate their intersection sizes
        # (i.e. how many feature values are common)
        for f in self.use_features:
            this_adjacency = get_article_feature_adjacency_matrix(input_entities, f)
            # Accumulating all boosting from all features into a single matrix
            similarity_enh += this_adjacency

        similarity_enh_clipped = np.clip(similarity_enh, 0, self.ftr_max_energy)
        similarity_enh_clipped = similarity_enh_clipped / self.ftr_max_energy

        adjacency_matrix = np.zeros_like(similarity_enh)
        adjacency_matrix[similarity_enh >= self.ftr_th] = 1
        # obtain subgraphs based on connected components
        g = nx.Graph(adjacency_matrix)
        sub_graphs = nx.connected_components(g)

        # for every subgraph with high node numbers obtain eign vectors
        # the node number is a hyperparameter which can be optimized
        overall_data = pd.DataFrame()
        for graph in sub_graphs:
            if len(graph) >= self.graph_node_th:
                spectral_entities = [input_entities[i] for i in graph]
                sub_similarity_matrix = [similarity_enh_clipped[i] for i in graph]
                overall_graph, eign_val, eign_vctrs = spectral.get_graph_eign(
                    spectral_entities, self.use_features
                )
                graph_df = spectral.get_node_eign_vector(
                    overall_graph, spectral_entities, eign_vctrs, 1, 4
                )
                # get graph similarity matrix
                spec_similarity_enh, final_graph, spec_sub_graphs = spectral.get_graph_clusters(
                    graph_df, self.th
                )
                mapped_spec_similarity = []
                for i in range(len(spec_similarity_enh)):
                    spec_df = pd.DataFrame(
                        {
                            "orig": list(graph),
                            "new": list(spectral_entities),
                            "spectal_val": spec_similarity_enh[i],
                        }
                    )
                    sub_df = pd.DataFrame([{"sub_spectal_val": sub_similarity_matrix[i]}])
                    final_df = pd.merge(
                        sub_df, spec_df, how="left", left_on=sub_df.index, right_on=spec_df.orig
                    )
                    final_df = final_df.fillna(0)
                    mapped_spec_similarity.append(final_df.spectal_val.values)
                # the steps are the matrix manipulation steps to match
                mapped_spec_similarity = np.array(mapped_spec_similarity)
                mapped_spec_similarity = np.clip(
                    mapped_spec_similarity, self.min_energy, self.max_energy
                )
                mapped_df = pd.DataFrame(
                    {
                        "orig": list(graph),
                        "new": list(spectral_entities),
                        "spectal_val": list(mapped_spec_similarity),
                    }
                )
                sub_similarity_df = pd.DataFrame({"sim_val": list(similarity_enh_clipped)})
                all_vector_df = pd.merge(
                    sub_similarity_df,
                    mapped_df,
                    how="left",
                    left_on=sub_similarity_df.index,
                    right_on=mapped_df.orig,
                )

                overall_data = overall_data.append(all_vector_df[["key_0", "spectal_val"]])
        all_node_vectors = []
        for i, data_vector in enumerate(overall_data.index.unique()):
            vector_val_arr = overall_data[overall_data.key_0 == i]["spectal_val"].values
            final_vector = [val for val in vector_val_arr if type(val) != float]
            if len(final_vector) > 0:
                node_vector = final_vector[0]
            else:
                node_vector = np.array([0])

            all_node_vectors.append(node_vector)

        if len(all_node_vectors) == len(similarity_enh_clipped):
            similarity_featr_graph = similarity_enh_clipped + all_node_vectors
        else:
            similarity_featr_graph = similarity_enh_clipped + np.zeros(
                similarity_enh_clipped.shape, dtype=float
            )
        sim += similarity_featr_graph
        return sim
