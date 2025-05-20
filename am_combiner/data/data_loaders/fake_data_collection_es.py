# The script gets data based on a set of sampled random names and saves it by letter
import os
import random
from typing import List

import click
import pandas as pd
from elasticsearch import Elasticsearch

from am_combiner.utils.ab_utils import read_config, es_connect, get_document_es

PATH = "am_combiner/data/data_loaders/random_name_data/"
ALL_NAMES: pd.DataFrame = pd.read_csv("am_combiner/data/Validation Data Name Samples - Sheet3.csv")
ALL_NAMES["first_letter"] = ALL_NAMES.apply(lambda row: row["name"][0].lower(), axis=1)


def summarise_name_distribution(
    all_names: pd.DataFrame = ALL_NAMES, sample_size: int = 1000
) -> pd.DataFrame:
    """
    Summarise the distribution of names.

    Summarise by first letter and select samples from every set of names as per the distribution of
    names and overall sample size.

    Parameters
    ----------
    all_names:
        The pandas dataframe containing the names.
    sample_size:
        The sample size needed from the set of names.

    Returns
    -------
        A dataframe containing the distribution of names by first letter of alphabet and the samples
        based on their sample size.

    """
    names_by_first_letter = pd.DataFrame(
        all_names.groupby("first_letter")["name"].count().reset_index()
    )
    names_by_first_letter["prob"] = names_by_first_letter.apply(
        lambda row: row["name"] / names_by_first_letter["name"].sum(), axis=1
    )
    names_by_first_letter["new_sample"] = names_by_first_letter.apply(
        lambda row: round(row["prob"] * sample_size), axis=1
    )
    # the dataframe is sorted as it will help in batching the script so that user can select any or
    # some of the 26 alphabets by indexing
    names_by_first_letter = names_by_first_letter.sort_values("first_letter")
    return names_by_first_letter


def get_articles_by_name(entity_name: str) -> pd.DataFrame:
    """
    Get all the articles from ES, for a certain entity name.

    Parameters
    ----------
    entity_name:
        The entity name for which we need to extract the articles.

    Returns
    -------
        A pandas dataframe containing the articles.

    """
    # creating the elastic search connection
    config: dict = read_config()
    es_client: Elasticsearch = es_connect(config, "prod")

    # for an entity name get all the mentions from the elastic search
    mentions = get_document_es(
        es_client=es_client,
        config=config,
        query={"entity_name": entity_name},
        fields=["uri", "entity_name", "extracted_date", "listing_type", "listing_subtypes"],
        index_type="mentions",
        date_field_name="extracted_date",
    )
    listing_types_by_url = dict()
    for mention in mentions:
        listing_types_by_url[mention["uri"]] = (
            mention["listing_type"],
            mention["listing_subtypes"],
        )
    entity_urls: List = list(listing_types_by_url.keys())
    all_article_examples = pd.DataFrame()
    # for all the urls obtained fetch the articles divide the entire url set into batches of 50
    # so that batches are easy to maintain and consumes less memory
    # 50 is hardcoded as its been easy and less time consuming to collect the data
    for i in range(0, len(entity_urls), 50):
        urls = entity_urls[i : i + 50]
        query = {"url": urls}
        article_documents = get_document_es(
            es_client=es_client,
            config=config,
            query=query,
            fields=["domain", "date", "country", "language", "url", "title", "content"],
            index_type="articles",
            date_field_name="tstamp",
        )
        for document in article_documents:
            listing_type_and_subtypes = listing_types_by_url[document["url"]]
            document["listing_type"] = listing_type_and_subtypes[0]
            document["listing_subtype"] = listing_type_and_subtypes[1][0]["subtype"]

        all_article_examples = all_article_examples.append(pd.DataFrame(article_documents))
    all_article_examples["entity_name"] = entity_name

    return all_article_examples


def get_all_random_name_articles_by_letter(
    letter: str = "x", no_of_names_selected: int = 2
) -> pd.DataFrame:
    """
    Get all the articles from ES, for a set of random names taken by certain letters.

    Parameters
    ----------
    letter:
        The letter based on which the names are to be sampled.
    no_of_names_selected:
        The no of names which are to be sampled.

    Returns
    -------
        A dataframe which contains the articles for all the names for a particular letter.

    """
    all_random_name_articles = pd.DataFrame()
    random_names_by_letter = random.sample(
        list(ALL_NAMES[ALL_NAMES.first_letter == letter]["name"]), no_of_names_selected
    )
    for name in random_names_by_letter:
        print(name)
        all_articles_by_name = get_articles_by_name(name)
        all_random_name_articles = all_random_name_articles.append(all_articles_by_name)

    return all_random_name_articles


@click.command()
@click.option(
    "--letter",
    default="x",
    required=True,
    help="the first letter based on which the names are to be selected",
    type=str,
)
@click.option(
    "--sample_size",
    default=1000,
    required=True,
    help="the sample size of total names that we need to extract",
    type=int,
)
def get_all_random_name_all_articles(letter: str, sample_size: int) -> pd.DataFrame:
    """
    Get the letter and the sample size and then get all the articles and store them as pickle.

    To reduce load to ES and parallelize the extraction we chose to extract the articles by letter.

    Parameters
    ----------
    letter:
        Contains the letter for which we want to get the articles.
    sample_size:
        The overall sample size of names which needs to be extracted.

    Returns
    -------
        Data frame containing all articles for all random names selected as per the list of letter.

    """
    all_random_name_all_articles = pd.DataFrame()
    name_distribution_by_letter = summarise_name_distribution(sample_size=sample_size)
    print(letter)
    no_of_names_selected = int(
        name_distribution_by_letter[name_distribution_by_letter.first_letter == letter][
            "new_sample"
        ]
    )
    all_random_name_articles = get_all_random_name_articles_by_letter(letter, no_of_names_selected)
    all_random_name_all_articles = all_random_name_all_articles.append(all_random_name_articles)

    file_path = os.path.join(PATH, f"all_random_name_articles_{letter}.pickle")
    all_random_name_all_articles.to_pickle(file_path)

    return all_random_name_articles


if __name__ == "__main__":
    get_all_random_name_all_articles()
