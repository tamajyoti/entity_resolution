from typing import Tuple, List, Dict, Optional

import pandas as pd
from sklearn import metrics

from am_combiner.combiners.common import (
    CLUSTER_NUMBER_FIELD,
    CLUSTER_ID_FIELD,
    ENTITY_NAME_CLUSTER_ID_FIELD,
    CLUSTER_SUPPORT_FIELD,
    IS_OVER_FIELD,
    IS_UNDER_FIELD,
    NAME_OC_RATE_FIELD,
    NAME_UC_RATE_FIELD,
    PROFILES_PER_OC_FIELD,
    PROFILES_CREATED_FIELD,
    PROFILES_TRUE_FIELD,
    SCORE_TO_MINIMISE_FIELD,
    NAME_FIELD,
    HOMOGENEITY_FIELD,
    COMPLETENESS_FIELD,
    V_SCORE_FIELD,
    COUNT_FIELD,
    BLOCKING_FIELD_FIELD,
    UNIQUE_ID_FIELD,
    GROUND_TRUTH_FIELD,
)
from am_combiner.qa.quality_control import ClusteringQualityReporter
from am_combiner.qa.utils import get_expected_and_actual_clustering
from am_combiner.qa.acceptance_metrics import get_acceptance_scores


def validate_combiner(
    validation_df: pd.DataFrame, clustering_results_df: pd.DataFrame, verbose: Optional[bool] = True
) -> Tuple[Dict, pd.DataFrame]:
    """
    Validate combiner and output the results.

    Parameters
    ----------
    validation_df:
        pd.DataFrame containing the clustering ground truth.
    clustering_results_df:
        pd.DataFrame containing the clustering results.
    verbose:
        If true, print the final validation results.

    Returns
    -------
        The generated overall report and the dataframe containing the validation results per name.

    """
    name_counts = perform_initial_name_checks(validation_df, clustering_results_df)

    clustering_list, quality_list = list(), list()
    for entity_name, count in name_counts.iteritems():
        try:
            expected_clustering, actual_clustering = get_expected_and_actual_clustering(
                entity_name, validation_df, clustering_results_df
            )
        except ValueError:
            print(f"Will not evaluate {entity_name}")
            continue

        clustering, quality = validate_name(entity_name, expected_clustering, actual_clustering)

        clustering_list.extend(clustering)
        quality_list.append(quality)

    clustering_df = pd.DataFrame(clustering_list)
    quality_df = pd.DataFrame(quality_list).set_index(NAME_FIELD)

    if verbose:
        print_validation_results(clustering_df, quality_df)

    quality_reporter = ClusteringQualityReporter(clustering_quality_df=quality_df)
    report = quality_reporter.clustering_report(weights_column=COUNT_FIELD, verbose=verbose)

    return report, quality_df


def perform_initial_name_checks(
    validation_df: pd.DataFrame, clustering_results_df: pd.DataFrame
) -> pd.Series(dtype=str):
    """
    Perform initial basic checks on the clustering dataframe, such as if all names are present etc.

    Parameters
    ----------
    clustering_results_df:
        pd.DataFrame containing the clustering results.
    validation_df:
        pd.DataFrame containing the clustering ground truth.

    Returns
    -------
        A Pandas dataframe containing the name counts.

    """
    if (
        not clustering_results_df[BLOCKING_FIELD_FIELD]
        .isin(validation_df[BLOCKING_FIELD_FIELD])
        .all()
    ):
        print("Not all clusterting blocks are presented in the validation set.")

    validation_name_counts = validation_df[BLOCKING_FIELD_FIELD].value_counts()
    names_counts_df = clustering_results_df[BLOCKING_FIELD_FIELD].value_counts().rename("Count")

    for block_name, actual_count in names_counts_df.iteritems():
        if block_name not in validation_name_counts:
            print(f"Skipping {block_name} as it is not presented in the validation set")
            continue

        validation_count = validation_name_counts[block_name]
        if validation_count != actual_count:
            print(
                f"Entity occurrence counts for validation and labeled set do not match: "
                f"{block_name}: {validation_count}(validation) vs {actual_count}(actual)"
            )

    return names_counts_df


def validate_name(
    entity_name: str, expected_clustering: pd.DataFrame, actual_clustering: pd.DataFrame
) -> Tuple[List[Dict], Dict]:
    """
    Validate combiner results for a name.

    Parameters
    ----------
    entity_name:
        The entity name to validate combiner results for.
    expected_clustering:
        The expected clustering results (ground truth).
    actual_clustering:
        The actual clustering results.

    Returns
    -------
        A tuple of the clustering for that name and the quality scores for that name.

    """
    name_clustering = get_clustering_per_name(entity_name, expected_clustering, actual_clustering)

    homogeneity, completeness, v_score = get_homogeneity_completeness_v_score(
        expected_clustering, actual_clustering
    )
    oc_rate, uc_rate, profiles_per_oc, profiles_created, true_profiles = get_oc_uc_scores(
        expected_clustering, name_clustering
    )
    score_to_minimise = get_score_to_minimise(
        oc_rate, uc_rate, profiles_per_oc, profiles_created, true_profiles
    )

    name_quality = {
        NAME_FIELD: entity_name,
        HOMOGENEITY_FIELD: homogeneity,
        COMPLETENESS_FIELD: completeness,
        V_SCORE_FIELD: v_score,
        COUNT_FIELD: len(actual_clustering),
        NAME_OC_RATE_FIELD: oc_rate,
        NAME_UC_RATE_FIELD: uc_rate,
        PROFILES_PER_OC_FIELD: profiles_per_oc,
        PROFILES_CREATED_FIELD: profiles_created,
        PROFILES_TRUE_FIELD: true_profiles,
        SCORE_TO_MINIMISE_FIELD: score_to_minimise,
    }

    return name_clustering, name_quality


def get_homogeneity_completeness_v_score(
    expected_clustering: pd.DataFrame, actual_clustering: pd.DataFrame
) -> Tuple[float, float, float]:
    """
    Get the homogeneity, completeness and V score results, for a name.

    Parameters
    ----------
    expected_clustering:
        The expected clustering results (ground truth).
    actual_clustering:
        The actual clustering results.

    Returns
    -------
        A tuple of floats representing the homogeneity, completeness and V score.

    """
    homogeneity, completeness, v_score = metrics.homogeneity_completeness_v_measure(
        labels_true=expected_clustering[GROUND_TRUTH_FIELD].to_list(),
        labels_pred=actual_clustering[CLUSTER_NUMBER_FIELD].to_list(),
    )

    return round(homogeneity, 2), round(completeness, 2), round(v_score, 2)


def get_clustering_per_name(
    block_name: str, expected_clustering: pd.DataFrame, actual_clustering: pd.DataFrame
) -> List[Dict]:
    """
    Get the clustering results, for a name.

    Parameters
    ----------
    block_name:
        The block name to get clustering results for.
    expected_clustering:
        The expected clustering results (ground truth).
    actual_clustering:
        The actual clustering results.

    Returns
    -------
        A list of dictionaries, each dictionary representing information for a cluster.

    """
    clustering_list = list()
    ac_grouped = actual_clustering.set_index([CLUSTER_ID_FIELD, UNIQUE_ID_FIELD])
    ec_grouped = expected_clustering.set_index([GROUND_TRUTH_FIELD, UNIQUE_ID_FIELD])
    for actual_cluster_number in actual_clustering[CLUSTER_NUMBER_FIELD].unique():
        actual_urls = set(ac_grouped.loc[actual_cluster_number].index)

        mask = expected_clustering[UNIQUE_ID_FIELD].isin(actual_urls)
        expected_clusters = expected_clustering.loc[mask, GROUND_TRUTH_FIELD].unique()

        missing_urls = list()
        for expected_cluster_number in expected_clusters:
            expected_urls = set(ec_grouped.loc[expected_cluster_number].index)
            missing_urls.extend(expected_urls.difference(actual_urls))

        clustering_list.append(
            {
                BLOCKING_FIELD_FIELD: f"{block_name}",
                ENTITY_NAME_CLUSTER_ID_FIELD: f"{block_name}-MVP{actual_cluster_number}",
                CLUSTER_SUPPORT_FIELD: len(expected_clusters),
                IS_UNDER_FIELD: len(missing_urls) > 0,
                IS_OVER_FIELD: len(expected_clusters) > 1,
            }
        )

    return clustering_list


def get_oc_uc_scores(
    expected_clustering: pd.DataFrame, name_clustering: List[Dict]
) -> Tuple[float, float, float, int, int]:
    """
    Get over and under combination scores, for a name.

    Parameters
    ----------
    expected_clustering:
        The expected clustering results (ground truth).
    name_clustering:
        The actual clustering of the name for which to get the scores.

    Returns
    -------
        A tuple representing the OC rate, UC rate, profiles per OC, the number of profiles created
        and the number of true profiles.

    """
    profiles_created = len(name_clustering)

    is_over_no = sum([cluster[IS_OVER_FIELD] for cluster in name_clustering])
    oc_rate = round(is_over_no / profiles_created, 2)

    is_under_no = sum([cluster[IS_UNDER_FIELD] for cluster in name_clustering])
    uc_rate = round(is_under_no / profiles_created, 2)

    oc_profiles = [
        cluster[CLUSTER_SUPPORT_FIELD] for cluster in name_clustering if cluster[IS_OVER_FIELD]
    ]
    profiles_per_oc = round(sum(oc_profiles) / len(oc_profiles), 2) if oc_profiles else 0

    true_profiles = expected_clustering[GROUND_TRUTH_FIELD].unique().size

    return oc_rate, uc_rate, profiles_per_oc, profiles_created, true_profiles


def get_score_to_minimise(
    oc_rate: float,
    uc_rate: float,
    profiles_per_oc: float,
    profiles_created: int,
    true_profiles: int,
) -> float:
    """
    Get the score to minimise, or product score, for a name.

    Parameters
    ----------
    oc_rate:
        Over combination rate per name.
    uc_rate:
        Under combination rate per name.
    profiles_per_oc:
        Number of profiles per over combination per name.
    profiles_created:
        Number of profiles created per name.
    true_profiles:
        Number of profiles which should have been created per name (ground truth).

    Returns
    -------
        A float value representing the score to minimise.

    """
    score_to_minimise = (
        3.4 * uc_rate
        + 8.4 * oc_rate
        + 0.4 * profiles_per_oc
        + 0.8 * max(profiles_created - true_profiles, 0) / true_profiles
    )

    return round(score_to_minimise, 2)


def print_validation_results(clustering_df: pd.DataFrame, quality_df: pd.DataFrame):
    """
    Print final validation results.

    Parameters
    ----------
    clustering_df:
        The actual clustering results.
    quality_df:
        The quality scores of the clustering.

    """
    print(clustering_df)
    print(quality_df)

    p_uc = clustering_df[IS_UNDER_FIELD].sum() / len(clustering_df)
    p_oc = clustering_df[IS_OVER_FIELD].sum() / len(clustering_df)

    print(f"Percentage UC: {round(p_uc * 100, 2)}%")
    print(f"Percentage OC: {round(p_oc * 100, 2)}%")


def check_acceptance_distribution(
    cluster_df: pd.DataFrame,
    sampling_rate: int = 0.2,
    number_of_runs: int = 20,
):
    """Check acceptance distribution based on mutliple runs."""
    all_scores = pd.DataFrame()
    for i in range(number_of_runs):
        accuracy, precision, recall, fscore = get_acceptance_scores(cluster_df, sampling_rate)
        all_scores = all_scores.append(
            pd.DataFrame(
                [
                    {
                        "run": i,
                        "accuracy": round(accuracy, 2),
                        "precision": round(precision, 2),
                        "recall": round(recall, 2),
                        "fscore": round(fscore, 2),
                    }
                ]
            )
        )
    return all_scores
