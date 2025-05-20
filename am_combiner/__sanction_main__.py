from pathlib import Path

import click
import pandas as pd
import json

from am_combiner.blockers.helpers import get_blockers_cache
from am_combiner.defaults import (
    DEFAULT_MONGO_DATABASE,
    DEFAULT_MO_COLLECTION,
    DEFAULT_PROFILE_COLLECTION,
    DEFAULT_SM_COLLECTION,
    DEFAULT_TEST_PROPORTION,
    DEFAULT_VALIDATION_PROPORTION,
    DEFAULT_READ_SAVED,
    DEFAULT_RESULT_STORAGE,
    DEFAULT_CACHED_INPUT,
    DEFAULT_CACHE_PATH,
    DEFAULT_INCLUDED_SPLITS,
    DEFAULT_DUMP_VISITED_RECORDS,
    DEFAULT_IGNORE_VALIDATION,
    DEFAULT_SM_TYPES,
    DEFAULT_COMBINE_INTER_DOMAIN,
    DEFAULT_BLOCKER,
    DEFAULT_FULL_DATASET,
)
from am_combiner.utils.sanction_data import ManualOverlayUnifyGroundTruth
from am_combiner.features.frontend import SanctionFeatureExtractorFrontend
from am_combiner.features.helpers import get_visitors_cache
from am_combiner.utils.parametrization import features_str_to_enum, get_cache_from_yaml
from am_combiner.combiners.mapping import COMBINER_CLASS_MAPPING
from am_combiner.combiners.common import (
    combine_entities_wrapper,
    TRAIN_TEST_VALIDATE_SPLIT_FIELD,
    UNIQUE_ID_FIELD,
    GROUND_TRUTH_FIELD,
    CLUSTER_ID_FIELD,
    BLOCKING_FIELD_FIELD,
)
from am_combiner.qa.quality_metrics import validate_combiner
from am_combiner.utils.storage import STORAGE_MAPPING, OutputUriKeeper
from am_combiner.splitters.common import load_splitter

DEFAULT_MONGO_URI = (
    "mongodb://"
    "mongodb-staging-001.euw1.data-staging-1,"
    "mongodb-staging-002.euw1.data-staging-1,"
    "mongodb-staging-003.euw1.data-staging-1"
    "/company"
)


@click.command(context_settings={"show_default": True})
@click.option(
    "--mongo-uri",
    help="Mongo URI for whatever reasons you may need it",
    default=DEFAULT_MONGO_URI,
    expose_value=True,
)
@click.option("--mongo-database", help="mongo db name", default=DEFAULT_MONGO_DATABASE)
@click.option("--mo-collection", help="Manual Overlay collection", default=DEFAULT_MO_COLLECTION)
@click.option(
    "--profile-collection", help="Primary profiles collection", default=DEFAULT_PROFILE_COLLECTION
)
@click.option("--sm-collection", help="Sanctions collection", default=DEFAULT_SM_COLLECTION)
@click.option(
    "--test-prop", help="Proportion of annotation used for testing", default=DEFAULT_TEST_PROPORTION
)
@click.option(
    "--valid-prop",
    help="Proportion of annotation used for testing",
    default=DEFAULT_VALIDATION_PROPORTION,
)
@click.option(
    "--read-saved", help="If True, download pickled sanctions from S3.", default=DEFAULT_READ_SAVED
)
@click.option("--visitors", multiple=True, help="Define visitors to be run on the sanction")
@click.option(
    "--combiners", is_eager=True, default=False, multiple=True, help="Which combiner to use"
)
@click.option("--splitter", default=None, help="Which splitter apply after all combiners")
@click.option(
    "--results-storage",
    type=click.Choice([DEFAULT_RESULT_STORAGE]),
    default=DEFAULT_RESULT_STORAGE,
    help="Specify the flag if the results need to be stored in mongo, rather than locally",
)
@click.option(
    "--output-path",
    is_eager=True,
    required=True,
    help="Absolute path to the csv file containing output clustering results",
    type=click.Path(),
)
@click.option(
    "--entity-types",
    is_eager=True,
    multiple=True,
    help="Restrict the list of names to be processed. "
    "Use it if you need to process a subset of names",
)
@click.option(
    "--cached-input",
    help="True, if we require reading the input df form a cached inputs." "False, otherwise.",
    type=click.BOOL,
    default=DEFAULT_CACHED_INPUT,
)
@click.option(
    "--cache-path",
    is_eager=True,
    required=True,
    help="Absolute path to the cache folder",
    type=click.Path(),
    default=DEFAULT_CACHE_PATH,
)
@click.option(
    "--included-splits",
    multiple=True,
    help="subset of annotated data.",
    type=click.Choice(["test", "valid", "train"]),
    default=DEFAULT_INCLUDED_SPLITS,
)
@click.option(
    "--dump-visited-records",
    help="If True, pickled visited records will be dumped into output directory",
    type=click.BOOL,
    default=DEFAULT_DUMP_VISITED_RECORDS,
)
@click.option(
    "--ignore-validation",
    help="If True, the combining results will not be validated, only saved.",
    type=click.BOOL,
    default=DEFAULT_IGNORE_VALIDATION,
)
@click.option(
    "--sm-types",
    help="type of structured mention",
    multiple=True,
    type=click.Choice(["sanction", "pep-class-1", "pep-class-2", "pep-class-3", "pep-class-4"]),
    default=DEFAULT_SM_TYPES,
)
@click.option(
    "--full-dataset",
    help="Either return full dataset or filter for only manually overwritten primary profiles.",
    type=bool,
    default=DEFAULT_FULL_DATASET,
)
@click.option(
    "--combine-inter-domain",
    help="If True, the inter-domain combination will be performed.",
    type=click.BOOL,
    default=DEFAULT_COMBINE_INTER_DOMAIN,
)
@click.option(
    "--blocker",
    help="The name of the blocker to use.",
    multiple=False,
    default=DEFAULT_BLOCKER,
)
def main(
    mongo_uri,
    sm_collection,
    mongo_database,
    mo_collection,
    profile_collection,
    test_prop,
    valid_prop,
    read_saved,
    visitors,
    combiners,
    splitter,
    results_storage,
    output_path,
    entity_types,
    cache_path,
    cached_input,
    included_splits,
    dump_visited_records,
    sm_types,
    full_dataset,
    ignore_validation,
    blocker,
    combine_inter_domain,
):
    """

    Wrap the main function.

    The reason we have the wrapper so that we could call the original
    function from both command line and as a python function.

    """
    main_(
        mongo_uri,
        sm_collection,
        mongo_database,
        mo_collection,
        profile_collection,
        test_prop,
        valid_prop,
        read_saved,
        visitors,
        combiners,
        splitter,
        results_storage,
        output_path,
        entity_types,
        cache_path,
        cached_input,
        included_splits,
        dump_visited_records,
        sm_types,
        full_dataset,
        ignore_validation,
        blocker,
        combine_inter_domain,
    )


def main_(
    mongo_uri=DEFAULT_MONGO_URI,
    sm_collection=DEFAULT_SM_COLLECTION,
    mongo_database=DEFAULT_MONGO_DATABASE,
    mo_collection=DEFAULT_MO_COLLECTION,
    profile_collection=DEFAULT_PROFILE_COLLECTION,
    test_prop=DEFAULT_TEST_PROPORTION,
    valid_prop=DEFAULT_VALIDATION_PROPORTION,
    read_saved=DEFAULT_READ_SAVED,
    visitors=None,
    combiners=None,
    splitter=None,
    results_storage=DEFAULT_RESULT_STORAGE,
    output_path=None,
    entity_types=None,
    cache_path=DEFAULT_CACHE_PATH,
    cached_input=DEFAULT_CACHED_INPUT,
    included_splits=DEFAULT_INCLUDED_SPLITS,
    dump_visited_records=DEFAULT_DUMP_VISITED_RECORDS,
    sm_types=DEFAULT_SM_TYPES,
    full_dataset=DEFAULT_FULL_DATASET,
    ignore_validation=DEFAULT_IGNORE_VALIDATION,
    blocker=DEFAULT_BLOCKER,
    combine_inter_domain=DEFAULT_COMBINE_INTER_DOMAIN,
) -> OutputUriKeeper:
    """

    Process sanctions input data, loop through combiners and validate results.

    Main entry point for sanctions.

    """
    storage_params = {}
    if results_storage == DEFAULT_RESULT_STORAGE:
        storage_params["output_path"] = output_path

    storage_saver_class = STORAGE_MAPPING[results_storage]
    storage_saver = storage_saver_class(**storage_params)

    keeper = OutputUriKeeper()
    keeper.output_path = output_path

    sanction_df = None
    cache_full_path = Path(cache_path) / f"cached_{'_'.join(sorted(sm_types))}.csv"
    if cached_input and cache_full_path.exists():
        with open(cache_full_path, "r") as f:
            sanction_df = pd.DataFrame(json.load(f))

    if sanction_df is None:
        provider = ManualOverlayUnifyGroundTruth(
            params={
                "mongo_uri": mongo_uri,
                "mongo_database": mongo_database,
                "sm_collection": sm_collection,
                "mo_collection": mo_collection,
                "profile_collection": profile_collection,
                "test_prop": test_prop,
                "valid_prop": valid_prop,
                "read_saved": read_saved,
                "entity_types": entity_types,
                "sm_types": sm_types,
                "full_dataset": full_dataset,
            }
        )
        sanction_df = provider.get_dataframe()
        if cache_path:
            with open(cache_full_path, "w+") as f:
                f.write(sanction_df.to_json())

    sm_types_suffix = "-".join(sm_types)
    dataframe_uri = f"{sm_types_suffix}-dataframe.csv"
    keeper._dataframe_uri = dataframe_uri
    storage_saver.store_dataframe(df=sanction_df, uri=dataframe_uri)
    sanction_df = sanction_df[sanction_df[TRAIN_TEST_VALIDATE_SPLIT_FIELD].isin(included_splits)]

    config_path = "sanction_combiners_config.yaml"
    visitors_cache = get_visitors_cache(visitors=visitors, config_path=config_path)
    entity_sanctions = SanctionFeatureExtractorFrontend(
        visitors_cache=visitors_cache, visitors=visitors, thread_count=4
    ).produce_visited_objects_from_df(input_entities_df=sanction_df)

    if blocker:
        blocker_object = get_blockers_cache(blocker_name=blocker, config_path=config_path)
        print(f"Blocking data with {blocker}")
        entity_sanctions = blocker_object.block_data(entity_sanctions)
        print(f"Done blocking. Largest block: {max([len(v) for v in entity_sanctions.values()])}")

    if dump_visited_records:
        import pickle

        with open(Path(cache_path) / "visited_objects.pkl", "wb") as f:
            pickle.dump(entity_sanctions, f)

    splitter_obj = load_splitter(splitter, config_path)
    combiner_cache = get_cache_from_yaml(
        config_path,
        section_name="combiners",
        class_mapping=COMBINER_CLASS_MAPPING,
        restrict_classes=combiners,
        attrs_callbacks={
            "source_feature": features_str_to_enum,
            "node_features": features_str_to_enum,
            "use_features": lambda fs: [features_str_to_enum(f) for f in fs],
        },
    )
    reports = []
    for ct, combiner in enumerate(combiners):
        print(f"Combining entities with {combiner} ({ct + 1}/{len(combiners)})")
        clustering_results_df, average_time_by_mention_no = combine_entities_wrapper(
            entity_articles=entity_sanctions,
            combiner_object=combiner_cache[combiner],
            splitter=splitter_obj,
        )
        if blocker:
            clustering_results_df = blocker_object.deblock_labels(clustering_results_df)
        clustering_results_df = pd.merge(
            clustering_results_df,
            sanction_df[[UNIQUE_ID_FIELD, BLOCKING_FIELD_FIELD, GROUND_TRUTH_FIELD]],
            on=[UNIQUE_ID_FIELD, BLOCKING_FIELD_FIELD],
            how="left",
        )
        results_dataframe_uri = f"{combiner}-{sm_types_suffix}-results.csv"
        keeper._results_dataframe_uri = results_dataframe_uri
        storage_saver.store_dataframe(df=clustering_results_df, uri=results_dataframe_uri)
        if combine_inter_domain:
            print("Starting inter-domain combination...")
            print("Done inter-domain combination")
        if ignore_validation:
            continue
        unique_clusters_ids = clustering_results_df[GROUND_TRUTH_FIELD].unique()
        ids_counts = []
        for cid in unique_clusters_ids:
            mask = clustering_results_df[GROUND_TRUTH_FIELD] == cid
            output_cid = clustering_results_df.loc[mask, CLUSTER_ID_FIELD]

            sids = clustering_results_df.loc[mask, UNIQUE_ID_FIELD]
            q = f"match (s:SANCTION_ID)-[]-(e:DATA) where s.value in {[s for s in sids]} return s,e"
            ids_counts.append({"cid": cid, "ids count": len(output_cid.unique()), "q": q})
            clustering_results_df.loc[mask, "Q"] = q
        output_unique_cids = clustering_results_df[CLUSTER_ID_FIELD].unique()
        oc_ids_counts = []
        for cid in output_unique_cids:
            mask = clustering_results_df[CLUSTER_ID_FIELD] == cid
            output_cid = clustering_results_df.loc[mask, GROUND_TRUTH_FIELD]
            sids = clustering_results_df.loc[mask, UNIQUE_ID_FIELD]
            q = f"match (s:SANCTION_ID)-[]-(e:DATA) where s.value in {[s for s in sids]} return s,e"
            oc_ids_counts.append({"cid": cid, "ids count": len(output_cid.unique()), "q": q})

        storage_saver.store_dataframe(
            df=pd.DataFrame(ids_counts), uri=f"{combiner}-{sm_types_suffix}-cluster-counts-uc.csv"
        )
        storage_saver.store_dataframe(
            df=pd.DataFrame(oc_ids_counts),
            uri=f"{combiner}-{sm_types_suffix}-cluster-counts-oc.csv",
        )
        report, clustering_quality_df = validate_combiner(sanction_df, clustering_results_df)
        report["combiner"] = combiner
        reports.append(report)

    if reports:
        report_frame = pd.DataFrame(reports).set_index("combiner")
        storage_saver.store_dataframe(
            df=report_frame, uri=f"all-combiners-results-{sm_types_suffix}.csv"
        )
        print(report_frame.transpose())

    return keeper


if __name__ == "__main__":
    main()
