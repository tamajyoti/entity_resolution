from am_combiner.combiners.fastRP import FastRPCosineSim
from am_combiner.features.sanction import Sanction
import numpy as np
import pytest


@pytest.mark.parametrize(
    "sanctions",
    [
        [Sanction("SM:HilaryClinton47", {}, "person"), Sanction("SM:DonaldTrump46", {}, "person")],
    ],
)
def test_permutation_invariance(sanctions):
    combiner = FastRPCosineSim(use_features=[])

    R1 = combiner._compute_deterministic_random_projection_matrix(sanctions)
    R2 = combiner._compute_deterministic_random_projection_matrix(sanctions[::-1])

    np.testing.assert_array_equal(R1.toarray()[0, :], R2.toarray()[1, :])
    np.testing.assert_array_equal(R1.toarray()[1, :], R2.toarray()[0, :])

    assert R1.shape[0] == len(sanctions)
    assert R1.shape[1] == combiner.dim


@pytest.mark.parametrize(
    ("A",),
    ((np.array([[0, 0, 0], [0, 0, 0.5], [0, 0.5, 0]]),),),
)
def test_fast_rp_cosine(A):
    combiner = FastRPCosineSim(use_features=[])
    np.random.seed(42)
    R = np.random.rand(A.shape[0], 100)
    sim = combiner._fastrp_proj(A, R)
    # Check that node 1 and 2 are similar, while 0 is not.
    assert sim[0, 1] < sim[1, 2]
    assert sim[0, 2] < sim[1, 2]
