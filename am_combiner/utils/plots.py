from pathlib import Path
from typing import List, Dict

import pandas as pd
from matplotlib import pyplot

from am_combiner.utils.storage import AbstractResultsSaver


def plot_sensitivity_analysis_histograms(
    combiner: str,
    histograms: List[str],
    sensitivity_analysis: pd.DataFrame,
    holdout_ration: float,
    storage_saver: AbstractResultsSaver,
):
    """
    Plot and save sensitivity analysis histograms to file.

    Parameters
    ----------
    storage_saver:
        I/O wrapper implementation
    combiner:
        Combiner to plot sensitivity analysis results for.
    histograms:
        List of histograms to plot (for example V score).
    sensitivity_analysis:
        The sensitivity analysis results.
    holdout_ration:
        The fraction of the sample.

    """
    for hist in histograms:
        if hist not in sensitivity_analysis:
            print(f"Metric {hist} does not exist in sensitivity analysis result")
            continue

        holdout_pp = int(holdout_ration * 100)
        storage_saver.store_histogram_input(
            histogram_input={
                "data": sensitivity_analysis[hist],
                "title": f'{combiner} "{hist}" {holdout_pp}% hold',
            },
            uri=f"{combiner} {hist} {holdout_pp}% hold.png",
        )


def plot_time_performance_histograms(
    combiner: str, average_time_by_mention_no: Dict[int, list], path: Path
):
    """
    Plot and save time performance histogram to file.

    Parameters
    ----------
    combiner:
        Combiner to plot time performance histogram for.
    average_time_by_mention_no:
        Dictionary of average time to cluster a name by number of mentions.
    path:
        The path where to save the histograms.

    """
    path.mkdir(parents=True, exist_ok=True)

    pyplot.figure()
    pyplot.bar(average_time_by_mention_no.keys(), average_time_by_mention_no.values())
    pyplot.xlabel("Number of mentions")
    pyplot.ylabel("Time (ms) to cluster a name")
    pyplot.savefig(path / f"{combiner}-time-performance.png")
    pyplot.close()
