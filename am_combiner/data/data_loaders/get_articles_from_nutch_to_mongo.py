import csv
import time
from collections import defaultdict
from typing import List

import click
import pandas as pd
from pymongo import MongoClient

from am_combiner.data.data_loaders.fake_data_collection_es import get_articles_by_name

ACCEPTED_V2_LISTING_SUBTYPES = [
    "adverse-media-v2-property",
    "adverse-media-v2-financial-aml-cft",
    "adverse-media-v2-fraud-linked",
    "adverse-media-v2-narcotics-aml-cft",
    "adverse-media-v2-violence-aml-cft",
    "adverse-media-v2-terrorism",
    "adverse-media-v2-cybercrime",
    "adverse-media-v2-general-aml-cft",
    "adverse-media-v2-regulatory",
    "adverse-media-v2-financial-difficulty",
    "adverse-media-v2-violence-non-aml-cft",
    "adverse-media-v2-other-financial",
    "adverse-media-v2-other-serious",
    "adverse-media-v2-other-minor",
]


def get_names_from_mongo(
    mongo_uri: str,
    mongo_db_name_commonness: str,
    mongo_collection_name_commonness: str,
    commonness_rank: float,
) -> List[str]:
    """Get names from mongo by name commonness rank."""
    mongo_client = MongoClient(mongo_uri)
    mongo_collection = mongo_client[mongo_db_name_commonness][mongo_collection_name_commonness]

    names = list()
    for name in mongo_collection.find({"common_rank": {"$gte": commonness_rank}}):
        names.append(name["name"])

    print(f"Got {len(names)} names from Mongo with common rank greater than {commonness_rank}")

    return names


def get_all_articles(
    names: List[str],
    csv_already_fetched_names: str,
    get_by_commonness: bool,
    min_articles: int,
    max_articles: int,
    max_names: int,
) -> pd.DataFrame:
    """Get and return all articles for all the names in the list."""
    start_time = time.time()

    already_saved_names = get_names_from_csv(csv_already_fetched_names)

    checked_names = 0
    finished_names = 0

    articles = pd.DataFrame()
    for name in names:
        print(f"Will start checking {name}")
        checked_names += 1

        if name in already_saved_names:
            continue

        start_name = time.time()

        try:
            name_articles = get_articles_by_name(name)
        except Exception as ex:
            print(f"Got exception {ex}")
            continue

        print(f"Checked {checked_names} / {len(names)} names after {time.time() - start_time}s")

        name_articles = name_articles.drop_duplicates(subset=["url"])
        name_articles = name_articles[name_articles.language == "en"]
        name_articles = name_articles[name_articles.listing_type != "pep"]
        name_articles = name_articles[
            name_articles.listing_subtype.isin(ACCEPTED_V2_LISTING_SUBTYPES)
        ]

        if get_by_commonness:
            if not min_articles <= len(name_articles) <= max_articles:
                continue

        articles = articles.append(name_articles)
        finished_names += 1

        print(
            f"Finished getting articles for {name}, found {len(name_articles)} unique URLS "
            f"in {time.time() - start_name}s"
        )

        if max_names and finished_names >= max_names:
            break

    print(f"Got articles for {finished_names} names")

    return articles


def dump_all_articles_to_mongo(
    mongo_uri: str,
    mongo_db_to_save: str,
    mongo_collection_to_save: str,
    articles_to_dump: pd.DataFrame,
):
    """Dump all the articles in a Mongo collection."""
    mongo_client = MongoClient(mongo_uri)

    articles_to_dump.rename(columns={"_id": "index_id"}, inplace=True)
    articles_to_dump = articles_to_dump.reset_index(drop=True)
    articles_to_dump["_id"] = articles_to_dump.reset_index().index

    mongo_client[mongo_db_to_save][mongo_collection_to_save].insert_many(
        articles_to_dump.to_dict("records"), ordered=False, bypass_document_validation=True
    )


def get_statistics_about_articles(
    mongo_uri: str,
    mongo_db_to_save: str,
    mongo_collection_to_save: str,
    csv_currently_fetched_names: str,
):
    """Print statistics about how many articles each name has."""
    mongo_client = MongoClient(mongo_uri)
    mongo_collection = mongo_client[mongo_db_to_save][mongo_collection_to_save]

    mentions_by_name = defaultdict(list)
    for mention in mongo_collection.find():
        mentions_by_name[mention["entity_name"]].append(mention["url"])

    for name, urls in mentions_by_name.items():
        print(f"For {name} there are {len(urls)} unique urls")

    if csv_currently_fetched_names:
        dump_names_to_csv(list(mentions_by_name.keys()), csv_currently_fetched_names)


def get_names_from_csv(file_name: str) -> List[str]:
    """Get a list of names from a CSV file."""
    file_name = f"am_combiner/data/data_loaders/{file_name}"

    names = list()
    with open(file_name, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader, None)
        for row in csv_reader:
            if row:
                names.append(row[0])

    return names


def dump_names_to_csv(names: List[str], file_name: str):
    """Dump names to a CSV file."""
    file_name = f"am_combiner/data/data_loaders/{file_name}"

    with open(file_name, "w") as csv_file:
        dict_writer = csv.DictWriter(csv_file, fieldnames=["Name"])
        dict_writer.writeheader()
        for name in names:
            dict_writer.writerow({"Name": name})


@click.command()
@click.option(
    "--mongo-uri",
    default="mongodb://mongodb-mf-person-0-0.mi-playground-1"
    "/?replicaSet=mf-person-0&readPreference=primary&appname=MongoDB%20Compass&ssl=false",
    required=True,
    help="Mongo URI",
    type=str,
)
@click.option(
    "--mongo-db-to-save",
    default="am_combiner",
    required=True,
    help="Mongo database, where to save fetched articles",
    type=str,
)
@click.option(
    "--mongo-collection-to-save",
    required=True,
    help="Mongo collection, where to save fetched articles",
    type=str,
)
@click.option(
    "--mongo-db-name-commonness",
    default="name_commonness",
    required=False,
    help="Mongo database, where to fetch name commonness information from",
    type=str,
)
@click.option(
    "--mongo-collection-name-commonness",
    default="all_name_commonness",
    required=False,
    help="Mongo collection, where to fetch name commonness information from",
    type=str,
)
@click.option(
    "--csv-names-to-fetch",
    default="names_to_fetch.csv",
    required=False,
    help="CSV file, where to get names to fetch from (can be None if get-by-commonness is set)",
    type=str,
)
@click.option(
    "--csv-currently-fetched-names",
    required=False,
    help="CSV file, where to save currently fetched names to",
    type=str,
)
@click.option(
    "--csv-already-fetched-names",
    default="already_fetched_names.csv",
    required=False,
    help="CSV file, where to get already fetched names fetched from",
    type=str,
)
@click.option(
    "--get-by-commonness",
    default=False,
    required=False,
    help="True, if articles should be fetched by commonness instead of a list of names",
    type=bool,
)
@click.option(
    "--commonness-rank",
    default=1e-07,
    required=False,
    help="If get-by-commonness is True, the commonness rank of the names",
    type=float,
)
@click.option(
    "--min-articles",
    default=60,
    required=False,
    help="If get-by-commonness is True, the minimum number of articles a name should have",
    type=int,
)
@click.option(
    "--max-articles",
    default=300,
    required=False,
    help="If get-by-commonness is True, the maximum number of articles a name should have",
    type=int,
)
@click.option(
    "--max-names",
    required=False,
    help="If get-by-commonness is True, the maximum number of names to be fetched",
    type=int,
)
def main(
    mongo_uri: str,
    mongo_db_to_save: str,
    mongo_collection_to_save: str,
    mongo_db_name_commonness: str,
    mongo_collection_name_commonness: str,
    csv_names_to_fetch: str,
    csv_currently_fetched_names: str,
    csv_already_fetched_names: str,
    get_by_commonness: bool,
    commonness_rank: float,
    min_articles: int,
    max_articles: int,
    max_names: int
):
    """Run main function to get articles from Nutch to Mongo."""
    start = time.time()

    if get_by_commonness:
        names_to_fetch = get_names_from_mongo(
            mongo_uri, mongo_db_name_commonness, mongo_collection_name_commonness, commonness_rank
        )
    else:
        names_to_fetch = get_names_from_csv(csv_names_to_fetch)

    articles_to_dump = get_all_articles(
        names_to_fetch,
        csv_already_fetched_names,
        get_by_commonness,
        min_articles,
        max_articles,
        max_names
    )
    dump_all_articles_to_mongo(
        mongo_uri, mongo_db_to_save, mongo_collection_to_save, articles_to_dump
    )

    if get_by_commonness:
        get_statistics_about_articles(
            mongo_uri, mongo_db_to_save, mongo_collection_to_save, csv_currently_fetched_names
        )

    print("Total elapsed", time.time() - start)


if __name__ == "__main__":
    main()
