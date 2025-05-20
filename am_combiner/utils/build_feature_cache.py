import io
import pickle

import click
from pymongo import MongoClient
import numpy as np

from am_combiner.combiners.common import URL_FIELD
from am_combiner.config import read_mongo
from am_combiner.features.frontend import ArticleFeatureExtractorFrontend
from am_combiner.features.helpers import get_visitors_cache
from am_combiner.utils.random_data import preprocess_input_dataframe


@click.command(context_settings={"show_default": True})
@click.option(
    "--mongo-uri",
    is_eager=True,
    help="Mongo URI for whatever reasons you may need it",
    default="mongodb://"
    "mongodb-mf-person-0-0.mi-playground-1"
    "/?replicaSet=mf-person-0&readPreference=primary&appname=MongoDB%20Compass&ssl=false",
    expose_value=True,
)
@click.option(
    "--mongo-database",
    is_eager=True,
    help="The name of input database to be used as a combiner input",
    default="am_combiner",
)
@click.option(
    "--mongo-collection",
    is_eager=True,
    help="The name of input collection to be used as a combiner input",
    required=True,
    multiple=True,
)
@click.option(
    "--overwrite-existing",
    help="If True, existing cache is removed first",
    type=bool,
    default=False,
)
@click.option(
    "--cache-name",
    type=str,
    help="name new cache",
    default="cache",
)
@click.option(
    "--meta-data-keys",
    multiple=True,
    required=False,
    help="Define meta data keys to be fetched in dataframe",
)
@click.option("--visitors", multiple=True, help="Define visitors to be run on the article")
def main(
    mongo_uri,
    visitors,
    mongo_database,
    mongo_collection,
    overwrite_existing,
    cache_name,
    meta_data_keys,
):
    """Rebuild feature cache."""
    for mc in mongo_collection:
        print(f"Doing collection {mc}")
        mongo_client = MongoClient(mongo_uri)
        input_entities_df = read_mongo(mongo_client, mongo_database, mc, {}, {})

        input_entities_df = preprocess_input_dataframe(input_entities_df, meta_data_keys)
        requested_urls = input_entities_df[URL_FIELD].tolist()
        # Target cache collections
        cache_collection = MongoClient(mongo_uri)["er-feature-cache"][cache_name]
        urls_filter = {"url": {"$in": requested_urls}}
        if overwrite_existing:
            cache_collection.delete_many(urls_filter)
        # Filter our links that were already processed
        cursor = cache_collection.find(urls_filter, {"url": True})
        processed_urls = [d["url"] for d in cursor]
        to_be_processed = set(requested_urls) - set(processed_urls)
        if not to_be_processed:
            print(f"Collection {mc} has nothing to process, moving to the next one.")
            continue
        input_entities_df = input_entities_df[input_entities_df[URL_FIELD].isin(to_be_processed)]
        print(f"Total records to be processed: {len(input_entities_df)}")
        # Save every 300 records to the db
        all_dfs = np.array_split(input_entities_df, max(int(len(input_entities_df) / 300), 1))
        frontend = ArticleFeatureExtractorFrontend(
            visitors_cache=get_visitors_cache(visitors=visitors),
            visitors=visitors,
            thread_count=8,
        )
        for idx, df in enumerate(all_dfs):
            print(f"Doing {idx + 1} of {len(all_dfs)}")

            articles_dict = frontend.produce_visited_objects_from_df(input_entities_df=df)

            for name, articles in articles_dict.items():
                for a in articles:
                    f = io.BytesIO()
                    pickle.dump(a.extracted_entities, f)
                    f.seek(0)
                    extracted_entities = f.read()

                    cache_collection.insert_one(
                        {"url": a.url, "extracted_features": extracted_entities}
                    )


if __name__ == "__main__":
    main()
