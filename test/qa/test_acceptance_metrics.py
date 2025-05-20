from am_combiner.qa.acceptance_metrics import get_acceptance_scores, get_url_map
from am_combiner.qa.quality_metrics import check_acceptance_distribution
import pandas as pd

CLUSTERING_DF = pd.DataFrame(
    {
        "cluster_number": [0, 1],
        "entity_name": "FAKE_ENTITY",
        "url": ["http://www.newera.com.na/2015/08/24/", "http://www.newera.com.na/2015/08/25/"],
        "cluster_links": [
            "http://www.newera.com.na/2015/08/24/",
            "http://www.newera.com.na/2015/08/25/",
        ],
        "ClusterID": [0, 0],
    }
)


def test_acceptance_metrics():
    url_combinations = get_url_map(CLUSTERING_DF, 1)
    acceptance_score = get_acceptance_scores(CLUSTERING_DF, 1)

    assert len(url_combinations) == 2

    assert url_combinations[0].clus_match == "No"
    assert url_combinations[0].valid_match == "Yes"
    assert url_combinations[1].clus_match == "No"
    assert url_combinations[1].valid_match == "Yes"

    assert acceptance_score == (0, 0, 0, 0)


def test_check_acceptance_distribution():
    acceptance_distribution = check_acceptance_distribution(CLUSTERING_DF, 1, 2).reset_index(
        drop=True
    )

    pd.testing.assert_frame_equal(
        acceptance_distribution,
        pd.DataFrame(
            {
                "run": [0, 1],
                "accuracy": [0.0, 0.0],
                "precision": [0.0, 0.0],
                "recall": [0.0, 0.0],
                "fscore": [0.0, 0.0],
            }
        ),
    )
