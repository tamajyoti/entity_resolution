import math
from typing import Dict, List, Tuple

import pandas as pd

from am_combiner.combiners.common import (
    TEXT_COLUMN_FIELD,
    UNIQUE_ID_FIELD,
    BLOCKING_FIELD_FIELD,
    GROUND_TRUTH_FIELD,
)
from am_combiner.features.article import Article


def get_expected_and_actual_clustering(
    blocking_name: str, validation_df: pd.DataFrame, clustering_results_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get expected and actual clusterings, ordered by URL, for a name."""
    expected_clustering = validation_df[
        validation_df[BLOCKING_FIELD_FIELD] == blocking_name
    ].sort_values(UNIQUE_ID_FIELD)

    actual_clustering = clustering_results_df[
        clustering_results_df[BLOCKING_FIELD_FIELD] == blocking_name
    ].sort_values(UNIQUE_ID_FIELD)

    return expected_clustering, actual_clustering


def get_cluster_by_name_and_url(
    df: pd.DataFrame,
    blocking_name: str,
    unique_id: str,
    unique_id_field: str = UNIQUE_ID_FIELD,
    blocking_field: str = BLOCKING_FIELD_FIELD,
    ground_truth_field: str = GROUND_TRUTH_FIELD,
) -> int:
    """Get cluster number for a certain name and URL from a dataframe."""
    mask = (df[unique_id_field] == unique_id) & (df[blocking_field] == blocking_name)

    return df.loc[mask, ground_truth_field].iat[0]


def article_to_df_element(article: Article, validation_df: pd.DataFrame) -> Dict:
    """Transform an Article object to a Pandas dataframe element."""
    return {
        BLOCKING_FIELD_FIELD: article.entity_name,
        TEXT_COLUMN_FIELD: article.article_text,
        UNIQUE_ID_FIELD: article.url,
        GROUND_TRUTH_FIELD: get_cluster_by_name_and_url(
            validation_df, article.entity_name, article.url
        ),
    }


def calculate_improvements(
    improvements_against: List[str], report_frame: pd.DataFrame, combiners: List[str]
):
    """
    Calculate improvements against different combiners.

    Parameters
    ----------
    improvements_against:
        List of combiner names to calculate improvements against.
    report_frame:
        Report containing results for all combiners.
    combiners:
        The list of all combiners.

    Returns
    -------
        List of improvements.

    """
    if "all" in improvements_against:
        improvements_against = combiners

    improvements: List[Dict] = list()

    for reference in improvements_against:
        for combiner in combiners:
            improvement = {"reference": reference, "combiner": combiner}
            for col in report_frame.columns:
                try:
                    numerator = report_frame.loc[combiner][col]
                    denominator = report_frame.loc[reference][col]
                    if not numerator and not denominator:
                        frac = math.nan
                    elif not denominator:
                        frac = math.inf
                    else:
                        frac = numerator / denominator
                except TypeError:
                    print(
                        f"Could not evaluate improvement for {col} "
                        f"for pair {reference} vs {combiner}"
                    )
                    continue
                improvement[col] = round(frac, 2)
            improvements.append(improvement)

    return improvements
