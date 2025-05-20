import numpy as np
import pandas as pd
from typing import List, Optional
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, spdiags
from sklearn.metrics.pairwise import cosine_similarity
from am_combiner.combiners.common import (
    Combiner,
    CombinedObjectFeature,
    CombinedObject,
    get_unique_ids_and_blocking_fields,
)
from am_combiner.splitters.common import Splitter
from am_combiner.utils.adjacency import get_article_multi_feature_adjacency


class FastRPCosineSim(Combiner):

    """
    A concrete implementation of a combiner class.

    Does clustering based on the connected components approach and utilises cosine similarity
    for creating graph connections.

    Attributes
    ----------
    th: float
        Cosine similarity threshold.
    dim: int
        The dimension of resulting FastRP representation.

    """

    def __init__(
        self,
        use_features: List[CombinedObjectFeature],
        th: float = 0.5,
        dim: int = 128,
    ):
        super().__init__()
        self.th = th
        self.random_val = 0.658
        self.dim = dim
        self.proj_weights = [0, 0.5, 0.5]
        self.features = use_features

    def _compute_deterministic_random_projection_matrix(self, sanctions: List[CombinedObject]):
        """
        Create RP initialization, which is permutation invariant.

        Parameters
        ----------
        sanctions:
            List of sanctions.

        Returns
        -------
            sanction permutation invariant random matrix.

        """
        arrs = []
        for sanction in sanctions:

            # Create a hashing from sanction_id -> seed.
            seed = 0
            for i, char in enumerate(sanction.sanction_id):
                seed += 3 ** i + ord(char)
            np.random.seed(seed % 2 ** 32)
            arr = np.random.choice(
                [0, -self.random_val, self.random_val], size=self.dim, p=[2.0 / 3, 1.0 / 6, 1.0 / 6]
            )
            arrs.append(arr)

        R = np.stack(arrs)
        return csr_matrix(R)

    def _fastrp_proj(self, A: coo_matrix, R: np.array) -> coo_matrix:
        """
        Calculate FastRP embeddings.

        Parameters
        ----------
        A:
            adjacency matrix
        R:
            random projection matrix.

        Returns
        -------
            Cosine similarity of FastRP representations.

        """
        assert len(self.proj_weights) > 0
        n = A.shape[0]
        A = csc_matrix(A)
        normalizer = spdiags(np.squeeze(1.0 / csc_matrix.sum(A, axis=1)), 0, n, n)
        M = normalizer @ A

        N_current = R
        N = np.zeros_like(R)
        for weight in self.proj_weights:
            N_current = M @ N_current
            N += N_current * weight

        return cosine_similarity(N)

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
        self, input_entities: List[CombinedObject], splitter: Optional[Splitter] = None
    ) -> pd.DataFrame:
        """
        Combine a list of given entities into clusters.

        Parameters
        ----------
        input_entities:
            A list of entities objects to be combined.
        splitter:
            Splitter on resulting adjacency matrix.

        Returns
        -------
            A pd.DataFrame object with cluster ids assigned to all entities.

        """
        fast_rp_adj = get_article_multi_feature_adjacency(
            input_entities, self.features, inverse_degree=True
        )
        R = self._compute_deterministic_random_projection_matrix(input_entities)
        sim = self._fastrp_proj(fast_rp_adj, R=R)
        adjacency_matrix = self._get_adjacency_from_similarities(sim)
        cluster_ids = Combiner.compute_cluster_ids_from_adjacency_matrix(
            coo_matrix(adjacency_matrix), input_entities, splitter
        )
        unique_ids, blocking_names = get_unique_ids_and_blocking_fields(input_entities)
        return Combiner.return_output_dataframe(
            cluster_ids=cluster_ids,
            unique_ids=unique_ids,
            blocking_names=blocking_names,
        )
