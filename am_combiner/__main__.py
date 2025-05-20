import copy
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

import click
import pandas as pd
from tqdm import tqdm

from am_combiner.combiners.common import (
    UNIQUE_ID_FIELD,
    BLOCKING_FIELD_FIELD,
    GROUND_TRUTH_FIELD,
    combine_entities_wrapper,
)
from am_combiner.combiners.mapping import COMBINER_CLASS_MAPPING
from am_combiner.features.frontend import (
    ArticleFeatureExtractorFrontend,
    FromCacheFeatureExtractionFrontend,
)
from am_combiner.features.graph_data import GraphVisualizationDataBuilder
from am_combiner.features.helpers import get_visitors_cache
from am_combiner.qa.cross_validation import (
    get_link_sensitivity_subsample,
    get_name_sensitivity_analysis,
)
from am_combiner.qa.quality_metrics import validate_combiner, check_acceptance_distribution
from am_combiner.qa.utils import calculate_improvements
from am_combiner.utils.cli import OptionThatRequiresOthers
from am_combiner.utils.data import DATA_PROVIDERS_CLASS_MAPPING, ANNOTATION_COMBINER
from am_combiner.utils.distributions import (
    DATA_DISTRIBUTION_MAPPER,
    TRUE_PROFILES_DISTRIBUTION_MAPPER,
)
from am_combiner.utils.parametrization import features_str_to_enum, get_cache_from_yaml
from am_combiner.utils.plots import (
    plot_sensitivity_analysis_histograms,
    plot_time_performance_histograms,
)
from am_combiner.utils.storage import STORAGE_MAPPING


@click.command(context_settings={"show_default": True})
@click.option(
    "--input-csv",
    is_eager=True,
    default=None,
    help="Path to the csv file containing input data to be combined",
    type=click.Path(),
)
@click.option(
    "--combiners-config-yaml",
    is_eager=True,
    default=None,
    help="Path to a yaml config that is used to overwrite static combiners configuration defined"
    "in combiners_config.yaml",
    type=click.Path(exists=True),
)
@click.option(
    "--combiners", is_eager=True, default=False, multiple=True, help="Which combiner to use"
)
@click.option(
    "--improvements-against",
    default=[],
    multiple=True,
    help="Defines which combiners to compare to each other",
    required=False,
)
@click.option("--visitors", multiple=True, help="Define visitors to be run on the article")
@click.option(
    "--entity-names",
    is_eager=True,
    multiple=True,
    help="Restrict the list of names to be processed. "
    "Use it if you need to process a subset of names",
)
@click.option(
    "--excluded-entity-names",
    is_eager=True,
    multiple=True,
    help="Exclude the list of names to be processed. "
    "Use it if you need to process a subset of names",
)
@click.option(
    "--max-names",
    is_eager=True,
    type=int,
    help="Restrict the maximum number of names to be processed."
    "Use it if you need to process a subset of names",
)
@click.option(
    "--max-content-length",
    is_eager=True,
    default=50000,
    type=int,
    help="Restrict the maximum length of text to be processed."
    "Use it if you need to discard some extra large articles",
)
@click.option(
    "--min-content-length",
    is_eager=True,
    default=0,
    type=int,
    help="Restrict the minimum length of text to be processed."
    "Use it if you need to discard some extra large articles",
)
@click.option(
    "--mongo-uri",
    is_eager=True,
    help="Mongo URI for whatever reasons you may need it",
    default="mongodb://"
    "mongodb-mf-person-0-0.mi-playground-1,"
    "mongodb-mf-person-0-1.mi-playground-1,"
    "mongodb-mf-person-0-2.mi-playground-1"
    "/?replicaSet=mf-person-0&readPreference=primary&appname=MongoDB%20Compass&ssl=false",
    expose_value=True,
)
@click.option(
    "--mongo-collection",
    is_eager=True,
    multiple=True,
    help="The name of input collections to be used as a combiner input."
    "All but the first are ignored, if input is not set to random.",
    default="dqs_validation_set",
)
@click.option(
    "--mongo-database",
    is_eager=True,
    help="The name of input database to be used as a combiner input",
    default="am_combiner",
)
@click.option(
    "--thread-count",
    is_eager=True,
    type=int,
    help="The number of parallel processes to use when parsing the articles.",
    default=4,
)
@click.option(
    "--name-resamplings",
    is_eager=True,
    type=int,
    help="The number of cross-validation checks to be done at name level.",
    default=0,
)
@click.option(
    "--name-holdout-ratio",
    is_eager=True,
    type=click.FloatRange(0, 1),
    help="The fraction of the sample size to draw at name level cross-validation.",
    default=0.6,
)
@click.option(
    "--link-resamplings",
    is_eager=True,
    type=int,
    help="The number of cross-validation checks to be done at link level. "
    "It is different from sampling the output results, since changing "
    "the amount of links can in fact change the clustering results "
    "for a particular name.",
    default=0,
)
@click.option(
    "--link-holdout-ratio",
    is_eager=True,
    type=click.FloatRange(0, 1),
    help="The fraction of the sample size to draw at link level cross-validation.",
    default=0.6,
)
@click.option(
    "--global-link-resampling",
    is_eager=True,
    type=click.BOOL,
    help="If true, a fraction from the global list of links will be drawn. "
    "If false, a fraction from the list of links of each name will be drawn.",
    default=False,
)
@click.option(
    "--histograms",
    is_eager=True,
    type=str,
    multiple=True,
    default=["V score", "Score to minimize", "Name UC rate", "Name OC rate"],
    help="The list of quality metrics to build histograms for",
)
@click.option(
    "--random-input-size",
    is_eager=True,
    type=int,
    default=10,
    help="How many random identities should be generated.",
)
@click.option(
    "--feature-frontend",
    is_eager=True,
    default="FeatureExtractorFrontend",
    help="Path to the pickle file containing input data with pre-computed features",
    type=click.Choice(("FeatureExtractorFrontend", "FromCacheFeatureExtractionFrontend")),
)
@click.option(
    "--experiment-id",
    is_eager=True,
    default="default-experiment",
    help="Provide an experiment id in order not to override results from previous experiments",
    type=str,
)
@click.option(
    "--graph-output",
    is_eager=True,
    type=click.BOOL,
    default=False,
    help="Specify this flag, we need graph nodes and links data to be generated"
    "Ensure that GraphDataVisitor is passed in the combiners list",
)
@click.option(
    "--name-set-distribution-summarizer-class",
    is_eager=True,
    type=click.Choice(["A", "B", "C", "C5000"]),
    help="Number of mentions frequency information provider",
    default="A",
)
@click.option(
    "--true-profiles-distribution-summarizer-class",
    is_eager=True,
    type=click.Choice(["A", "B", "C", "C5000"]),
    help="Number of true profiles information provider",
    default="A",
)
@click.option(
    "--results-storage",
    type=click.Choice(["mongo", "local"]),
    default="local",
    help="Specify the flag if the results need to be stored in mongo, rather than locally",
)
@click.option(
    "--input-data-source",
    type=click.Choice(["mongo", "csv", "random", "annotation"]),
    required=True,
    help="Specify the type in input data source. "
    "Certain data types will require different types of CLI args.",
    cls=OptionThatRequiresOthers,
    required_params={
        "csv": ["input_csv"],
        "mongo": ["mongo_uri", "mongo_database", "mongo_collection"],
        "random": [
            "random_input_size",
            "name_set_distribution_summarizer_class",
            "true_profiles_distribution_summarizer_class",
            "mongo_uri",
            "mongo_database",
            "mongo_collection",
        ],
        "annotation": ["input_csv"],
    },
)
@click.option(
    "--mongo-output-database",
    is_eager=True,
    help="The name of the output database for results storage",
    default="er-k8s-cluster-results",
)
@click.option(
    "--mongo-output-collection",
    is_eager=True,
    help="The name of the output collection for results storage",
    default="run_results",
)
@click.option(
    "--mongo-cache-database",
    is_eager=True,
    help="The name of the cache database for results storage",
    default="er-feature-cache",
)
@click.option(
    "--mongo-cache-collection",
    is_eager=True,
    help="The name of the cache collection for results storage",
    default="cache",
)
@click.option(
    "--skip-validation",
    help="True, if the validation should be skipped, meaning that the combiners will only cluster "
    "the input entities, without checking the quality. False, otherwise.",
    type=bool,
    default=False,
)
@click.option(
    "--accuracy-check",
    help="True, if the random sampling of urls to be done to get accuracy metrics "
    "False, otherwise.",
    type=bool,
    default=False,
)
@click.option(
    "--sampling-rate",
    is_eager=True,
    type=float,
    default=0.2,
    help="Percentage of total urls to be sampled, 0.2 is 20%",
)
@click.option(
    "--number-of-runs",
    is_eager=True,
    type=int,
    default=20,
    help="How many time acceptance rates are to be generated for a combiner.",
)
@click.option(
    "--store-per-profile-output",
    help="Whether to store per profile output frames. Reduces space usage during random data runs.",
    type=click.BOOL,
    default=True,
)
@click.option(
    "--store-input-frame",
    help="Whether to store the input dataframe. Reduces space usage during random data runs.",
    type=click.BOOL,
    default=True,
)
@click.option(
    "--meta-data-keys",
    multiple=True,
    required=False,
    help="Define meta data keys to be fetched in dataframe",
)
def main(
    input_csv,
    output_path,
    combiners,
    visitors,
    entity_names,
    mongo_uri,
    mongo_collection,
    mongo_database,
    thread_count,
    max_names,
    max_content_length,
    min_content_length,
    excluded_entity_names,
    name_resamplings,
    name_holdout_ratio,
    link_resamplings,
    link_holdout_ratio,
    global_link_resampling,
    improvements_against,
    histograms,
    random_input_size,
    feature_frontend,
    experiment_id,
    graph_output,
    name_set_distribution_summarizer_class,
    true_profiles_distribution_summarizer_class,
    results_storage,
    mongo_output_database,
    mongo_output_collection,
    input_data_source,
    combiners_config_yaml,
    skip_validation,
    accuracy_check,
    sampling_rate,
    number_of_runs,
    mongo_cache_database,
    mongo_cache_collection,
    store_per_profile_output,
    store_input_frame,
    meta_data_keys,
):
    """
    Process input data, loop through combiners and validate results.

    Main entry point.

    """
    output_path = Path(output_path) / experiment_id
    output_path.mkdir(parents=True, exist_ok=True)

    storage_params = {}
    if results_storage == "local":
        storage_params["output_path"] = output_path
    elif results_storage == "mongo":
        storage_params["mongo_uri"] = mongo_uri
        storage_params["database"] = mongo_output_database
        storage_params["collection"] = mongo_output_collection

    storage_saver_class = STORAGE_MAPPING[results_storage]
    storage_saver = storage_saver_class(**storage_params)

    if not set(improvements_against).issubset(combiners + ("all",)):
        raise ValueError(
            "Combiners specified in improvement-against option "
            "must be a subset of the evaluated combiners"
        )

    if input_data_source == "csv":
        data_provider_name = "CSVDataProvider"
        params = {"input_csv": input_csv}
    elif input_data_source == "random":
        data_provider_name = "RandomDataProvider"
        params = {
            "random_input_size": random_input_size,
            "name_set_distribution_summarizer_class": DATA_DISTRIBUTION_MAPPER[
                name_set_distribution_summarizer_class
            ],
            "true_profiles_distribution_summarizer_class": TRUE_PROFILES_DISTRIBUTION_MAPPER[
                true_profiles_distribution_summarizer_class
            ],
            "tag": experiment_id,
            "mongo_uri": mongo_uri,
            "mongo_database": mongo_database,
            "mongo_collection": mongo_collection,
        }
    elif input_data_source == "mongo":
        data_provider_name = "MongoDataProvider"
        params = {
            "mongo_uri": mongo_uri,
            "mongo_database": mongo_database,
            "mongo_collection": mongo_collection[0],
        }
    elif input_data_source == "annotation":
        data_provider_name = "AnnotationsProvider"
        params = {
            "input_csv": input_csv,
            "mongo_uri": mongo_uri,
            "mongo_database": mongo_database,
            "mongo_collection": mongo_collection[0],
        }
    else:
        raise ValueError(f"Unknown input_data_source: {input_data_source}")

    meta_keys = meta_data_keys if meta_data_keys else ()
    params["meta_keys"] = meta_keys

    data_provider_class = DATA_PROVIDERS_CLASS_MAPPING[data_provider_name]
    provider = data_provider_class(
        params=params,
        entity_names=entity_names,
        excluded_entity_names=excluded_entity_names,
        max_names=max_names,
        min_content_length=min_content_length,
        max_content_length=max_content_length,
    )

    input_entities_df = provider.get_dataframe()
    if skip_validation or input_data_source == "annotation":
        input_entities_df[GROUND_TRUTH_FIELD] = -1

    input_entities_df_to_save = input_entities_df[
        [UNIQUE_ID_FIELD, BLOCKING_FIELD_FIELD, GROUND_TRUTH_FIELD]
    ]
    if store_input_frame:
        # The only reason why you might not want to store it
        # is to save a lot of space during random runs.
        storage_saver.store_dataframe(df=input_entities_df_to_save, uri="input_dataframe.csv")

    validation_df = copy.copy(input_entities_df)
    if validation_df is not None:
        group = validation_df.groupby(BLOCKING_FIELD_FIELD)
        storage_saver.store_histogram_input(
            histogram_input={
                "data": group[GROUND_TRUTH_FIELD].nunique(),
                "title": "Number of true profiles distribution",
            },
            uri="validation_data_cluster_num.png",
        )

    # If all combiners were requested, we have to load them all, with no
    # restrictions.
    take_all = False
    if "all" in combiners or combiners_config_yaml:
        take_all = True
        restrict_classes = []
    elif input_data_source == "annotation":
        restrict_classes = list(combiners).append(ANNOTATION_COMBINER)
    else:
        restrict_classes = combiners

    config_path = combiners_config_yaml or "combiners_config.yaml"
    combiner_cache = get_cache_from_yaml(
        config_path,
        section_name="combiners",
        class_mapping=COMBINER_CLASS_MAPPING,
        restrict_classes=restrict_classes,
        attrs_callbacks={
            "source_feature": features_str_to_enum,
            "node_features": features_str_to_enum,
            "use_features": lambda fs: [features_str_to_enum(f) for f in fs],
        },
    )
    if take_all:
        combiners = sorted(set(combiner_cache.keys()).difference(set([ANNOTATION_COMBINER])))

    start = time.time()

    entity_articles = []
    visitors_cache = None
    if feature_frontend == "FeatureExtractorFrontend":
        visitors_cache = get_visitors_cache(visitors=visitors, config_path=config_path)
        entity_articles = ArticleFeatureExtractorFrontend(
            visitors_cache=visitors_cache, visitors=visitors, thread_count=thread_count
        ).produce_visited_objects_from_df(input_entities_df=input_entities_df)
    elif feature_frontend == "FromCacheFeatureExtractionFrontend":
        entity_articles = FromCacheFeatureExtractionFrontend(
            mongo_uri=mongo_uri,
            cache_database=mongo_cache_database,
            cache_collection=mongo_cache_collection,
        ).produce_visited_objects_from_df(input_entities_df=input_entities_df)
    if not entity_articles:
        print("Empty entity set, nothing to process")
        return

    # For augmented clustering, calculate ground truth clusters after articles are created:
    if input_data_source == "annotation":
        validation_df = provider.get_ground_truth(
            input_entities_df, entity_articles, combiner_cache[ANNOTATION_COMBINER]
        )

    # Create all the graph data after article objects are generated for all the entities
    if graph_output:
        if visitors_cache and "GraphDataVisitor" in visitors_cache:
            graph_data_folder = Path(output_path) / "graph_data"
            graph_data_folder.mkdir(parents=True, exist_ok=True)
            for entity_name, input_articles in entity_articles.items():
                graph_data = GraphVisualizationDataBuilder.merge_article_graph_representations(
                    input_articles
                )

                graph_data_df = pd.DataFrame(
                    {
                        "name": [entity_name],
                        "articles": [len(input_articles)],
                        "graph_data": [graph_data],
                    }
                )
                with open(graph_data_folder / f"{entity_name}.pickle", "wb") as outfile:
                    pickle.dump(graph_data_df, outfile)
        else:
            print("GraphDataVisitor missing. Add to generate graph data")

    reports: List[Optional[Dict]] = []
    time_performance: Dict[str, float] = dict()

    for ct, combiner in enumerate(combiners):
        print(f"Combining entities with {combiner} ({ct + 1}/{len(combiners)})")
        clustering_results_df, average_time_by_mention_no = combine_entities_wrapper(
            entity_articles=entity_articles,
            combiner_object=combiner_cache[combiner],
        )

        time_performance[combiner] = round(
            sum(average_time_by_mention_no.values()) / len(average_time_by_mention_no.values()), 2
        )

        if skip_validation:
            continue

        # For now we only store timings locally.
        # TODO store timings in mongo
        if results_storage == "local":
            plot_time_performance_histograms(
                combiner, average_time_by_mention_no, output_path / "time-performance"
            )

        print(f"Evaluating quality for {combiner}")
        report, clustering_quality_df = validate_combiner(validation_df, clustering_results_df)
        clustering_results_df = pd.merge(
            clustering_results_df,
            validation_df[[UNIQUE_ID_FIELD, BLOCKING_FIELD_FIELD, GROUND_TRUTH_FIELD]],
            on=[UNIQUE_ID_FIELD, BLOCKING_FIELD_FIELD],
            how="left",
        )
        if store_per_profile_output:
            storage_saver.store_dataframe(df=clustering_results_df, uri=f"{combiner}_results.csv")

        if accuracy_check:
            acceptance_scores_df = check_acceptance_distribution(
                clustering_results_df, sampling_rate, number_of_runs
            )
            storage_saver.store_dataframe(
                df=acceptance_scores_df.reset_index(drop=True),
                uri=f"{combiner}-acceptance-rates.csv",
            )
        if name_resamplings:
            name_sensitivity_analysis = get_name_sensitivity_analysis(
                clustering_quality_df, name_resamplings, name_holdout_ratio
            )
            plot_sensitivity_analysis_histograms(
                combiner, histograms, name_sensitivity_analysis, name_holdout_ratio, storage_saver
            )

        if link_resamplings:
            print(f"Starting link level cross-validation for {combiner}")
            link_sensitivity_analysis = list()
            for _ in tqdm(range(link_resamplings)):
                entity_articles_subsample, validation_subsample = get_link_sensitivity_subsample(
                    entity_articles, validation_df, link_holdout_ratio, global_link_resampling
                )
                clustering_results_subsample_df, _ = combine_entities_wrapper(
                    entity_articles=entity_articles_subsample,
                    combiner_object=combiner_cache[combiner],
                    verbose=False,
                )
                report_subsample, _ = validate_combiner(
                    validation_subsample, clustering_results_subsample_df, verbose=False
                )
                link_sensitivity_analysis.append(report_subsample)
                plot_sensitivity_analysis_histograms(
                    combiner,
                    histograms,
                    pd.DataFrame(link_sensitivity_analysis),
                    link_holdout_ratio,
                    storage_saver,
                )
        if store_per_profile_output:
            storage_saver.store_dataframe(
                df=clustering_quality_df, uri=f"{combiner}-per-profile-quality.csv"
            )

        report["combiner"] = combiner
        reports.append(report)

    # reports can be empty, if no combiners were actually requested
    if reports:
        report_frame = pd.DataFrame(reports).set_index("combiner")
        storage_saver.store_dataframe(df=report_frame, uri="all_combiners_results.csv")
        print(report_frame.transpose())

        if improvements_against:
            improvements = calculate_improvements(improvements_against, report_frame, combiners)
            improvements_df = pd.DataFrame(improvements)
            # Make a composite index, to make a table looking more convenient for looking at
            improvements_df.set_index(["reference", "combiner"], inplace=True)
            storage_saver.store_dataframe(df=improvements_df, uri="all_combiners_improvements.csv")

    time_performance_df = pd.DataFrame(
        time_performance.items(), columns=["Combiner", "Average Time per Name (ms)"]
    )
    storage_saver.store_dataframe(df=time_performance_df, uri="all_combiners_time_performance.csv")

    print(f"Elapsed: {time.time() - start}")


if __name__ == "__main__":
    main()
