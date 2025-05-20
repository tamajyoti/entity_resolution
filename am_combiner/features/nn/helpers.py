from typing import Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from am_combiner.combiners.common import CLUSTER_ID_GLOBAL_FIELD, UNIQUE_ID_FIELD


def build_cluster_id_cache(input_train_entities: pd.DataFrame) -> Dict[str, int]:
    """

    Build cache for fast URL ClusterId lookups.

    Args:
    ----
    input_train_entities: pd.DataFrame
        Input dataframe. Dataframe must contain ClusterIdGlobal and url field names.

    Return:
    ------
    Dict[str, int]
        A one-to-one mapping between URLs and cluster ids.


    """
    if CLUSTER_ID_GLOBAL_FIELD not in input_train_entities.columns:
        raise ValueError(f"Given dataframe must contain {CLUSTER_ID_GLOBAL_FIELD} column")
    if UNIQUE_ID_FIELD not in input_train_entities.columns:
        raise ValueError(f"Given dataframe must contain {UNIQUE_ID_FIELD} column")
    cache = {}

    for ct, row in input_train_entities.iterrows():
        if (
            row[UNIQUE_ID_FIELD] in cache
            and cache[row[UNIQUE_ID_FIELD]] != row[CLUSTER_ID_GLOBAL_FIELD]
        ):
            raise ValueError(
                "Contradicting ids in the cache:"
                f"{cache[row[UNIQUE_ID_FIELD]]} vs {row[CLUSTER_ID_GLOBAL_FIELD]}"
                " for {row[URL_FIELD_NAME]}"
            )
        cache[row[UNIQUE_ID_FIELD]] = row[CLUSTER_ID_GLOBAL_FIELD]

    return cache


def binary_threshold_search(
    net,
    test_data_loader,
    test_df,
    calculate_score,
    t_min=0,
    t_max=1,
    hop_num=5,
    y_max=None,
    y_min=None,
    tried_x=[],
    tried_y=[],
):
    """
    Implement binary search for threshold that minimizes value of custom calculate_score() function.

    Parameters
    ----------
    net:
        trained GCN that is being evaluated
    test_data_loader:
        generator of test data
    test_df:
        data frame with ground truth cluster values.
    calculate_score
        function that retrieves score that is being minimized
    t_min
        lower threshold bound being searched
    t_max
        upper threshold bound being searched
    hop_num
        number of binary partitions to be calculated.
        Total number of calculate_score() calls is hop_num + 2.
    y_max
        calculate_score value for t_max
    y_min
        calculate_score value for t_min
    tried_x
        list of all threshold values
    tried_y
        corresponding list of all score values for the thresholds attempted

    Returns
    -------
    tried_x
        list of all threshold values
    tried_y
        corresponding list of all score values for the thresholds attempted

    """
    if hop_num == 0:
        return tried_x, tried_y

    if y_max is None:
        y_max = calculate_score(t_max, net, test_data_loader, test_df)
        tried_x.append(t_max)
        tried_y.append(y_max)

    if y_min is None:
        y_min = calculate_score(t_min, net, test_data_loader, test_df)
        tried_x.append(t_min)
        tried_y.append(y_min)

    t_middle = 0.5 * (t_min + t_max)
    y_middle = calculate_score(t_middle, net, test_data_loader, test_df)
    tried_x.append(t_middle)
    tried_y.append(y_middle)

    if y_min < y_max:
        return binary_threshold_search(
            net,
            test_data_loader,
            test_df,
            calculate_score,
            t_min=t_min,
            t_max=t_middle,
            hop_num=hop_num - 1,
            y_max=y_middle,
            y_min=y_min,
            tried_x=tried_x,
            tried_y=tried_y,
        )
    else:
        return binary_threshold_search(
            net,
            test_data_loader,
            test_df,
            calculate_score,
            t_min=t_middle,
            t_max=t_max,
            hop_num=hop_num - 1,
            y_max=y_max,
            y_min=y_middle,
            tried_x=tried_x,
            tried_y=tried_y,
        )


def get_auc(label1: str, label2: str, data_frame: pd.DataFrame):
    """Get the auc value based on input characteristices."""
    auc = []
    for epoch in data_frame.epoch.unique():
        x = data_frame[data_frame.epoch == epoch][label1]
        y = data_frame[data_frame.epoch == epoch][label2]

        auc.append((epoch, np.round(np.trapz(y, x), 4)))

    return auc


def get_plot(label1: str, label2: str, data_frame: pd.DataFrame):
    """Get the graphs for UC_OC and Homogeneity_Completeness for every epoch."""
    a4_dims = (20, 10)
    fig, ax = plt.subplots(figsize=a4_dims)

    palette = ["r", "b", "g"]

    p1 = sns.lineplot(
        x=label1,
        y=label2,
        hue="epoch",
        style="epoch",
        markers=False,
        dashes=False,
        data=data_frame,
        color=palette,
        linewidth=2.5,
    )

    ax.legend(get_auc(label1, label2, data_frame))

    p1.grid()

    return ax
