import numpy as np
import pytest
from scipy.sparse import coo_matrix

from am_combiner.blockers.country import CountryBlocker
from am_combiner.features.sanction import Sanction


@pytest.mark.parametrize(
    ["ids_by_country", "adj", "country_options", "expected_country"],
    [
        (
            {"UK": set([1, 2]), "GR": set([3, 4, 5]), "AL": set([7, 8])},
            coo_matrix((np.ones(3), (np.zeros(3), [3, 4, 7])), shape=(9, 9)),
            ["AL", "UK"],
            "AL",
        ),
        (
            {"UK": set([1, 2]), "GR": set([3, 4, 5]), "AL": set([7, 8])},
            coo_matrix((np.ones(3), (np.zeros(3), [3, 4, 7])), shape=(9, 9)),
            None,
            "GR",
        ),
    ],
)
def test_calculating_best_country(ids_by_country, adj, country_options, expected_country):
    # make adj symmetric:
    adj = coo_matrix(adj + adj.transpose())
    rs = [Sanction("", {}, "") for _ in range(adj.shape[0])]
    for cc, inxs in ids_by_country.items():
        for inx in inxs:
            rs[inx].extracted_entities[CountryBlocker.COUNTRY_FEATURE] = set([cc])

    blocker = CountryBlocker(use_features=["f"])
    resulting_cc = blocker._calculate_best_country(0, ids_by_country, adj, rs, country_options)
    assert resulting_cc == expected_country
