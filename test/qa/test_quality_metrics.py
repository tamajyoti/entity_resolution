from am_combiner.combiners.common import (
    NAME_FIELD,
    HOMOGENEITY_FIELD,
    COMPLETENESS_FIELD,
    V_SCORE_FIELD,
    COUNT_FIELD,
    NAME_OC_RATE_FIELD,
    NAME_UC_RATE_FIELD,
    PROFILES_PER_OC_FIELD,
    PROFILES_CREATED_FIELD,
    PROFILES_TRUE_FIELD,
    SCORE_TO_MINIMISE_FIELD,
    BLOCKING_FIELD_FIELD,
    ENTITY_NAME_CLUSTER_ID_FIELD,
    CLUSTER_SUPPORT_FIELD,
    IS_UNDER_FIELD,
    IS_OVER_FIELD,
)
from am_combiner.qa.quality_metrics import validate_name, validate_combiner


def test_validate_combiner(validation, clustering_results):
    report, quality_df = validate_combiner(validation, clustering_results)
    assert report == {
        HOMOGENEITY_FIELD: 0.92,
        COMPLETENESS_FIELD: 0.85,
        V_SCORE_FIELD: 0.88,
        NAME_OC_RATE_FIELD: 0.1,
        NAME_UC_RATE_FIELD: 0.4,
        PROFILES_PER_OC_FIELD: 1.0,
        PROFILES_CREATED_FIELD: 3.5,
        PROFILES_TRUE_FIELD: 3.0,
        SCORE_TO_MINIMISE_FIELD: 2.7,
    }
    assert quality_df.to_dict("list") == {
        HOMOGENEITY_FIELD: [0.84, 1.0],
        COMPLETENESS_FIELD: [0.70, 1.0],
        V_SCORE_FIELD: [0.76, 1.0],
        COUNT_FIELD: [7, 2],
        NAME_OC_RATE_FIELD: [0.2, 0.0],
        NAME_UC_RATE_FIELD: [0.8, 0.0],
        PROFILES_PER_OC_FIELD: [2.0, 0.0],
        PROFILES_CREATED_FIELD: [5, 2],
        PROFILES_TRUE_FIELD: [4, 2],
        SCORE_TO_MINIMISE_FIELD: [5.4, 0.0],
    }


def test_validate_name(entity_name, expected_clustering, actual_clustering):
    outputted_name_clustering, outputted_name_quality = validate_name(
        entity_name, expected_clustering, actual_clustering
    )
    assert outputted_name_clustering == [
        {
            BLOCKING_FIELD_FIELD: entity_name,
            ENTITY_NAME_CLUSTER_ID_FIELD: "Some Name-MVP1",
            CLUSTER_SUPPORT_FIELD: 2,
            IS_UNDER_FIELD: False,
            IS_OVER_FIELD: True,
        },
        {
            BLOCKING_FIELD_FIELD: entity_name,
            ENTITY_NAME_CLUSTER_ID_FIELD: "Some Name-MVP2",
            CLUSTER_SUPPORT_FIELD: 1,
            IS_UNDER_FIELD: True,
            IS_OVER_FIELD: False,
        },
        {
            BLOCKING_FIELD_FIELD: entity_name,
            ENTITY_NAME_CLUSTER_ID_FIELD: "Some Name-MVP3",
            CLUSTER_SUPPORT_FIELD: 1,
            IS_UNDER_FIELD: True,
            IS_OVER_FIELD: False,
        },
        {
            BLOCKING_FIELD_FIELD: entity_name,
            ENTITY_NAME_CLUSTER_ID_FIELD: "Some Name-MVP4",
            CLUSTER_SUPPORT_FIELD: 1,
            IS_UNDER_FIELD: True,
            IS_OVER_FIELD: False,
        },
        {
            BLOCKING_FIELD_FIELD: entity_name,
            ENTITY_NAME_CLUSTER_ID_FIELD: "Some Name-MVP5",
            CLUSTER_SUPPORT_FIELD: 1,
            IS_UNDER_FIELD: True,
            IS_OVER_FIELD: False,
        },
    ]

    assert outputted_name_quality == {
        NAME_FIELD: entity_name,
        HOMOGENEITY_FIELD: 0.84,
        COMPLETENESS_FIELD: 0.70,
        V_SCORE_FIELD: 0.76,
        COUNT_FIELD: 7,
        NAME_OC_RATE_FIELD: 0.2,
        NAME_UC_RATE_FIELD: 0.8,
        PROFILES_PER_OC_FIELD: 2,
        PROFILES_CREATED_FIELD: 5,
        PROFILES_TRUE_FIELD: 4,
        SCORE_TO_MINIMISE_FIELD: 5.4,
    }
