import math

import pandas as pd

from am_combiner.combiners.common import (
    TEXT_COLUMN_FIELD,
    HOMOGENEITY_FIELD,
    COMPLETENESS_FIELD,
    V_SCORE_FIELD,
    BLOCKING_FIELD_FIELD,
    UNIQUE_ID_FIELD,
    GROUND_TRUTH_FIELD,
)
from am_combiner.features.article import Article
from am_combiner.qa.utils import (
    get_expected_and_actual_clustering,
    get_cluster_by_name_and_url,
    article_to_df_element,
    calculate_improvements,
)


def test_get_expected_and_actual_clustering(
    entity_name, validation, clustering_results, expected_clustering, actual_clustering
):
    outputted_expected_clustering, outputted_actual_clustering = get_expected_and_actual_clustering(
        entity_name, validation, clustering_results
    )
    assert outputted_expected_clustering.to_dict("list") == expected_clustering.to_dict("list")
    assert outputted_actual_clustering.to_dict("list") == actual_clustering.to_dict("list")


def test_get_cluster_by_name_and_url(entity_name, validation):
    assert get_cluster_by_name_and_url(validation, entity_name, "url.1") == 1


def test_article_to_df_element(entity_name, validation):
    article = Article(entity_name=entity_name, url="url.1", article_text="Some text")

    assert article_to_df_element(article, validation) == {
        BLOCKING_FIELD_FIELD: entity_name,
        TEXT_COLUMN_FIELD: "Some text",
        UNIQUE_ID_FIELD: "url.1",
        GROUND_TRUTH_FIELD: 1,
    }


def test_calculate_improvements():
    first_combiner = "combiner_1"
    second_combiner = "combiner_2"
    combiners = [first_combiner, second_combiner]

    report_frame = pd.DataFrame(
        [
            {
                "combiner": first_combiner,
                HOMOGENEITY_FIELD: 0.4,
                COMPLETENESS_FIELD: 0.3,
                V_SCORE_FIELD: 0,
            },
            {
                "combiner": second_combiner,
                HOMOGENEITY_FIELD: 0.1,
                COMPLETENESS_FIELD: 0,
                V_SCORE_FIELD: 0,
            },
        ]
    ).set_index("combiner")

    improvements = calculate_improvements(["all"], report_frame, combiners)
    for improvement in improvements:
        for key, value in improvement.items():
            if type(value) == float:
                if math.isnan(value):
                    improvement[key] = "nan"
                elif math.isinf(value):
                    improvement[key] = "inf"

    assert improvements[0] == {
        "reference": "combiner_1",
        "combiner": "combiner_1",
        HOMOGENEITY_FIELD: 1.0,
        COMPLETENESS_FIELD: 1.0,
        V_SCORE_FIELD: "nan",
    }

    assert improvements[1] == {
        "reference": "combiner_1",
        "combiner": "combiner_2",
        HOMOGENEITY_FIELD: 0.25,
        COMPLETENESS_FIELD: 0.0,
        V_SCORE_FIELD: "nan",
    }

    assert improvements[2] == {
        "reference": "combiner_2",
        "combiner": "combiner_1",
        HOMOGENEITY_FIELD: 4.0,
        COMPLETENESS_FIELD: "inf",
        V_SCORE_FIELD: "nan",
    }

    assert improvements[3] == {
        "reference": "combiner_2",
        "combiner": "combiner_2",
        HOMOGENEITY_FIELD: 1.0,
        COMPLETENESS_FIELD: "nan",
        V_SCORE_FIELD: "nan",
    }
