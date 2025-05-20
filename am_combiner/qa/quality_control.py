from typing import Dict, Hashable

import numpy as np
import pandas as pd


class ClusteringQualityReporter:

    """
    A class for generating report on cluster quality.

    Attributes
    ----------
    clustering_quality_df: pd.DataFrame
        A dataframe containing clustering quality information, such as overcombine/undercombine
        rate, created profiles, score to minimise etc.

    """

    def __init__(self, clustering_quality_df: pd.DataFrame):
        self.clustering_quality_df = clustering_quality_df

    def clustering_report(self, weights_column: str = None, verbose: bool = True) -> Dict:
        """
        Generate a report on clustering quality.

        Parameters
        ----------
        weights_column:
            Weights column.
        verbose:
            If true, print the pair of metric label - metric value.

        Returns
        -------
            The generated report.

        """
        report: Dict = {}
        for label, content in self.clustering_quality_df.iteritems():
            if label == weights_column:
                continue
            average_label = f"{label}"
            average_metric = self.get_average_statistic(label, weights_column=None)

            # average_label_w = f"(w) {label}"
            # average_metric_w = self.get_average_statistic(label, weights_column=weights_column)
            if verbose:
                print(f"{average_label}: {average_metric}")
                # print(f"{average_label_w}: {average_metric_w}")
            report[average_label] = average_metric
            # report[average_label_w] = average_metric_w
        return report

    def get_average_statistic(self, column_name: Hashable, weights_column: str = None) -> float:
        """
        Get an average statistic given a column name.

        Parameters
        ----------
        column_name:
            The column for which to get the average statistics.

        weights_column:
            Weights column.

        Returns
        -------
            The average statistic.

        """
        if column_name not in self.clustering_quality_df.columns:
            raise ValueError(f"Column {weights_column} does not exist in the dataframe")
        if weights_column and weights_column not in self.clustering_quality_df.columns:
            raise ValueError(f"Weight column {weights_column} does not exist in the dataframe")

        use_these = np.invert(self.clustering_quality_df[column_name].isna())
        data = self.clustering_quality_df[column_name][use_these].to_list()
        weights = None
        if weights_column:
            weights = self.clustering_quality_df[weights_column][use_these].to_list()

        w_average = round(float(np.average(data, weights=weights)), 2)

        return w_average
