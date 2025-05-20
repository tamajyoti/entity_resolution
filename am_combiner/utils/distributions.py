import json
from collections import Counter

import pandas as pd

from am_combiner.combiners.common import ENTITY_NAME_FIELD, CLUSTER_ID_FIELD, URL_FIELD


class NameSetDistributionSummariser:

    """Summariser for the distribution of the number of mentions per name."""

    def __init__(self, hist_json_path: str):
        # The input dataframe is ignored on purpose.
        # What we are going to return does not depend on it, it only depends on the static state
        # of the set described in the tab `sample` of this documents
        # https://docs.google.com/spreadsheets/d/1c37Wdy8apXzEUKv_JTW6t0SVl3c_tVILcmwKOMlcQBs/edit#gid=1435850082
        # The document is filtered to only names from the Set A
        # (name commonness > 1.00E-07 and mentions <= 50)

        # Basically represents the histogram of the number of mentions per name
        raw_hist = json.load(open(hist_json_path, "r"))
        # By default, python json encoder converts keys into strings, so when we load json
        # there is no such thing as int key. We need to convert it back.
        self.number_of_mentions = {int(k): v for k, v in raw_hist.items()}


DATA_DISTRIBUTION_MAPPER = {
    "A": NameSetDistributionSummariser("am_combiner/data/distributions/name_set_a.json"),
    "B": NameSetDistributionSummariser("am_combiner/data/distributions/name_set_b.json"),
    "C": NameSetDistributionSummariser("am_combiner/data/distributions/name_set_c.json"),
    "C5000": NameSetDistributionSummariser("am_combiner/data/distributions/name_set_c_5000.json"),
}


class TrueProfilesDistribution:

    """Summarizes a distribution of the number of true profiles."""

    def __init__(self, hist_json_path: str):
        data = json.load(open(hist_json_path, "r"))
        self.true_profiles = data["true_profiles"]
        self.weights = data["weights"]


TRUE_PROFILES_DISTRIBUTION_MAPPER = {
    "A": TrueProfilesDistribution("am_combiner/data/distributions/name_set_a_true_profiles.json"),
    "B": TrueProfilesDistribution("am_combiner/data/distributions/name_set_b_true_profiles.json"),
    "C": TrueProfilesDistribution("am_combiner/data/distributions/name_set_c_true_profiles.json"),
    "C5000": TrueProfilesDistribution(
        "am_combiner/data/distributions/name_set_c_true_profiles.json"
    ),
}


class DataframeDistributionSummariser:

    """
    A dataframe distribution summariser helper class.

    It takes dataframe in the format of validation/input data and does some minor summarising
    about the cluster sizes distribution.

    Attributes
    ----------
    cluster_count_values: pd.Series
        Contains information about pairs of Names/Number of clusters for each name.
    cluster_counts_weights: collections.Counter
        Contains counters for each cluster size.
        Gives a general idea of how much more often some cluster sizes appear.
    weights_for_cluster_sizes: Dict[int, Dict[int, int]]
        For each cluster size contains summarising for distributions of the number of articles
        for each unique entity in a cluster.

    """

    def __init__(self, dataframe: pd.DataFrame):
        entity_name_grouping = dataframe.groupby(ENTITY_NAME_FIELD)
        self.cluster_count_values = pd.DataFrame(entity_name_grouping[CLUSTER_ID_FIELD].nunique())
        # Sorted is used here for debugging purposes, so that Counter appears to be sorted as well
        # Contains sampling weights for clusters, so that we could emulate the DQS distribution
        self.cluster_counts_weights = Counter(
            sorted(self.cluster_count_values[CLUSTER_ID_FIELD].tolist())
        )

        # Now we need to build distributions of number of articles for each cluster,
        # within one name
        weights_for_cluster_sizes = {}
        for entity_name, row in self.cluster_count_values.iterrows():
            entity_slice = dataframe[dataframe[ENTITY_NAME_FIELD] == entity_name]
            weights_for_cluster_sizes[row[CLUSTER_ID_FIELD]] = Counter(
                entity_slice.groupby(CLUSTER_ID_FIELD)[URL_FIELD].count()
            )
        self.weights_for_cluster_sizes = weights_for_cluster_sizes
