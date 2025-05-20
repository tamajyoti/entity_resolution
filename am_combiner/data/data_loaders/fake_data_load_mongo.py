import os
import re

import click
import pandas as pd
from pymongo import MongoClient

PATH: str = "am_combiner/data/data_loaders/random_name_data/"


@click.command()
@click.option(
    "--mongo-uri",
    help="Mongo URI for whatever reasons you may need it",
    default="mongodb://"
    "mongodb-mf-person-0-0.mi-playground-1"
    "/?replicaSet=mf-person-0&readPreference=primary&appname=MongoDB%20Compass&ssl=false",
)
@click.option(
    "--database",
    default="am_combiner",
    required=True,
    help="the name of the mongodb database",
    type=str,
)
@click.option(
    "--collection",
    default="random_validation_data_1K",
    required=True,
    help="the name of the collection",
    type=int,
)
def write_to_mongo(mongo_uri: str, database: str, collection: str) -> None:
    """
    Consume all the ES articles collected for the entire set of random names and update it in Mongo.

    Parameters
    ----------
    mongo_uri:
        The host for the mongo database.
    database:
        The name of the mongo database.
    collection:
        The name of the collection.

    """
    mongo_client = MongoClient(mongo_uri)

    all_random_articles = pd.DataFrame()
    files = [f for f in os.listdir(PATH) if re.match("all_random_name_articles_", f)]
    for file in files:
        random_name_data = pd.read_pickle(os.path.join(PATH, file))
        all_random_articles = all_random_articles.append(random_name_data)

    # Get only english articles and create a psuedo_id as "_id" to mitigate load conflict
    # due to Mongo Index

    all_random_articles_en = all_random_articles[all_random_articles.language == "en"]
    all_random_articles_en.rename(columns={"_id": "index_id"}, inplace=True)
    all_random_articles_en = all_random_articles_en.reset_index(drop=True)
    all_random_articles_en["_id"] = all_random_articles_en.reset_index().index

    mongo_client[database][collection].insert_many(
        all_random_articles_en.to_dict("records"), ordered=False, bypass_document_validation=True
    )


if __name__ == "__main__":
    write_to_mongo()
