from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import random


class UrlCombo:

    """A class of map all the combination of url sample."""

    def __init__(
        self,
        url_a: str,
        url_b: str,
        valid_id_a: int,
        valid_id_b: int,
        clus_id_a: int,
        clus_id_b: int,
    ):
        self.url_a = url_a
        self.url_b = url_b
        self.valid_id_a = valid_id_a
        self.valid_id_b = valid_id_b
        self.clus_id_a = clus_id_a
        self.clus_id_b = clus_id_b
        if clus_id_a == clus_id_b:
            self.clus_match = "Yes"
        else:
            self.clus_match = "No"
        if valid_id_a == valid_id_b:
            self.valid_match = "Yes"
        else:
            self.valid_match = "No"


def get_url_map(cluster_df: pd.DataFrame, sampling_rate: int = 0.2):
    """Create url combinations based on sampling rate."""
    no_of_combinations_created = int(len(cluster_df.url.unique()) * sampling_rate)
    url_combination_list = []
    for i in range(no_of_combinations_created):
        url_list = random.sample(list(cluster_df.url.unique()), 2)
        cluster_values = cluster_df[cluster_df.cluster_links.isin(url_list)].reset_index(drop=True)
        mention = UrlCombo(
            url_a=cluster_values.url[0],
            url_b=cluster_values.url[1],
            valid_id_a=cluster_values.ClusterID[0],
            valid_id_b=cluster_values.ClusterID[1],
            clus_id_a=cluster_values.cluster_number[0],
            clus_id_b=cluster_values.cluster_number[1],
        )
        url_combination_list.append(mention)
    return url_combination_list


def get_acceptance_scores(cluster_df: pd.DataFrame, sampling_rate: int = 0.2):
    """Calculate acceptance and other metrics based on random sample of url combinations."""
    name_url_combinations = []
    for i, name in enumerate(cluster_df.entity_name.unique()):
        cluster_set = cluster_df[cluster_df.entity_name == name].reset_index(drop=True)
        url_combinations = get_url_map(cluster_set, sampling_rate)
        name_url_combinations.append(url_combinations)
        all_combinations = [combo for url_combo in name_url_combinations for combo in url_combo]
    actuals = []
    predicted = []
    for combo in all_combinations:
        actuals.append(combo.valid_match)
        predicted.append(combo.clus_match)

    accuracy = accuracy_score(actuals, predicted, normalize=True)
    precision, recall, fscore, support = precision_recall_fscore_support(
        actuals, predicted, average="weighted"
    )

    return accuracy, precision, recall, fscore
