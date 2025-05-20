import math
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import numpy
import pandas as pd

from am_combiner.features.article import Article
from am_combiner.qa.quality_control import ClusteringQualityReporter
from am_combiner.qa.utils import article_to_df_element


def random_draw(sample: List[Any], holdout_ratio: float) -> List:
    """
    Draw a number of random elements from a sample.

    Parameters
    ----------
    sample:
        The sample to draw a random subsample from.
    holdout_ratio:
        The fraction of the sample.

    Returns
    -------
        A list of random elements from the original sample.

    """
    if holdout_ratio <= 0 or holdout_ratio >= 1:
        raise ValueError("holdout_ratio must be strictly between 0 and 1")

    subsample_size = math.ceil(len(sample) * holdout_ratio)
    subsample = numpy.random.choice(sample, subsample_size, replace=False)

    return subsample


def get_name_sensitivity_analysis(
    clustering_quality_df: pd.DataFrame, resamplings: int, holdout_ratio: float
) -> pd.DataFrame:
    """
    Perform the name sensitivity analysis.

    Parameters
    ----------
    clustering_quality_df:
        The complete clustering quality Pandas dataframe.
    resamplings:
        The number of resamplings to perform.
    holdout_ratio:
        The fraction of the sample.

    Returns
    -------
        A Pandas dataframe representing the analysis results.

    """
    unique_names = clustering_quality_df.index
    reports = []
    for _ in range(resamplings):
        sampled_names = random_draw(list(unique_names), holdout_ratio)
        sub_frame = clustering_quality_df[clustering_quality_df.index.isin(sampled_names)]
        report = ClusteringQualityReporter(sub_frame).clustering_report(verbose=False)
        reports.append(report)
    results = pd.DataFrame(reports)

    return results


def get_link_sensitivity_subsample(
    entity_articles: Dict[str, List[Article]],
    validation_df: pd.DataFrame,
    link_holdout_ratio: float,
    global_link_resampling: bool,
) -> Tuple[Dict[str, List[Article]], pd.DataFrame]:
    """
    Get a subsample for link level cross-validation.

    Parameters
    ----------
    entity_articles:
        Initial list of articles to take subsample from.
    validation_df:
        Initial validation to take subsample from.
    link_holdout_ratio:
        The fraction of the sample.
    global_link_resampling:
        Take subsample from global list of links or from list of links for each name.

    Returns
    -------
        A tuple of articles and validation subsample.

    """
    entity_articles_subsample = defaultdict(list)
    validation_subsample = list()

    if global_link_resampling:
        all_articles = [article for articles in entity_articles.values() for article in articles]
        random_articles = random_draw(all_articles, link_holdout_ratio)
        for random_article in random_articles:
            entity_articles_subsample[random_article.entity_name].append(random_article)
    else:
        for entity_name, articles in entity_articles.items():
            entity_articles_subsample[entity_name] = random_draw(articles, link_holdout_ratio)

    for articles in entity_articles_subsample.values():
        for article in articles:
            validation_subsample.append(article_to_df_element(article, validation_df))

    return entity_articles_subsample, pd.DataFrame(validation_subsample)
