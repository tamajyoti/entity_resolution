import json
import os
from typing import List, Dict, Tuple
import traceback
import logging

import click
import requests
import pandas as pd
from collator_stores.command_store import CommandStore
from collator_stores.entity_store import EntityStore, EntityLogEntryDAO
from entity_models import Source, Entity
from entity_models.convert import BaseSourceLookup, EntityVersionConverter
from pymongo import MongoClient

# The sanction data is truncated to have only the id and name
SANCTION_MONGO_URI = (
    "mongodb://"
    "mongodb-staging-001.euw1.data-staging-1"
)
SANCTION_DATABASE = "company"
SANCTION_COLLECTION = "sm_entity_log"
PRIMARY_HOST = "https://matching-live.k8s.euw1.search-staging-1primary/search"
MONGO_URL = "mongodb://mongodb-qa-01-001.euw1.data-playground-1"
COMBINER_HOST = "http://combiner.k8s.euw1.data-playground-1best/match/source_document"
PATH = "am_combiner/data/sanction_mapping/"


def get_sanction_data() -> List:
    """Connect to mongo db to get list of sanctions data."""
    collection = MongoClient(SANCTION_MONGO_URI)[SANCTION_DATABASE][SANCTION_COLLECTION]
    cursor = collection.find(
        {"entity.data.aml_types.aml_type": "sanction"},
        {"_id": False, "time_utc": False, "created_utc": False},
    )
    docs = [doc for doc in cursor]

    return docs


def get_primary_ids(name: str) -> List[str]:
    """Get primary ids associated with a name search in papi. with fixed fuzziness."""
    host_name = PRIMARY_HOST
    parameters = {
        "name": name,
        "fuzziness": 0.5,  # this are standard parameters hence hard coded
        "limit": 100,
    }
    results = requests.get(host_name, params=parameters, timeout=5).json()
    primary_ids = []
    for val in results["content"]["results"][0]["hits"]:
        primary_ids.append("P:" + val["doc"]["id"])

    return primary_ids


def entity_with_empty_sources(entity: Entity):
    """Get the entity object."""
    for index, entity_source in enumerate(entity.sources):
        entity_source.source = Source()
        entity.sources[index] = entity_source

    return entity


class EmptySourceLookup(BaseSourceLookup):

    """Class for conversion."""

    def source_id_by_scraper_id(self, scraper_id):
        """Convert source id helper function."""
        return None

    def scraper_by_source_id(self, source_id):
        """Convert scraper helper function."""
        return None, None


def get_converter() -> Tuple:
    """Convertor objects to convert mentions to proper format."""
    entity_version_converter = EntityVersionConverter(EmptySourceLookup())
    mongo_db = MongoClient(MONGO_URL).get_database()

    entity_store = EntityStore(
        entity_log_entry_dao=EntityLogEntryDAO(mongo_db), command_store=CommandStore(mongo_db)
    )

    return entity_version_converter, entity_store


def get_primary_documents(primary_ids: List[str], entity_store, entity_version_converter) -> List:
    """Get the primary documents in acceptable format for combiner input."""
    v1_primaries = []
    for primary_id in primary_ids:
        v2_primary = entity_store.get_one(primary_id)

        # For the primary entities, additional step to enhance the primary with empty sources
        # is needed because Collator does not know how to convert entities without sources to V1
        # So we need to enhance it with dummy (empty) sources
        v2_primary_with_empty_sources = entity_with_empty_sources(v2_primary)

        v1_primary = entity_version_converter.convert_to_v1(v2_primary_with_empty_sources)

        v1_primaries.append(v1_primary)

    return v1_primaries


def get_response(source_mention: Dict, primary_mentions: List[Dict]) -> Dict:
    """Get the likelihood of combination for a source mention to its corresponding primaries."""
    response = requests.get(
        url=COMBINER_HOST,
        data=json.dumps(
            {
                "source_document": source_mention,
                "primary_documents": primary_mentions,
                "top": 1,
            }
        ),
        headers={"content-type": "application/json"},
    )
    return response.json()


def get_source_mapping(
    source_mention_id: str, primary_ids: List[str], entity_version_converter, entity_store
) -> pd.DataFrame:
    """Get the mapping of source mention to primaries."""
    v2_source_mention = entity_store.get_one(source_mention_id)
    v1_source_mention = entity_version_converter.convert_to_v1(v2_source_mention)

    primary_documents = get_primary_documents(primary_ids, entity_store, entity_version_converter)

    response = get_response(v1_source_mention, primary_documents)
    temp_df = {
        "source_mention": source_mention_id,
        "primary_ids": response[0]["id"],
        "likelihood": response[0]["likelihood"],
    }

    return temp_df


@click.command()
@click.option("--start_index", default=0, prompt="Start index")
@click.option("--end_index", default=10, prompt="End index")
def get_mapping(start_index: int, end_index: int) -> pd.DataFrame:
    """Get the mapping for each source mention to their primaries."""
    sanction_docs = get_sanction_data()
    entity_version_converter, entity_store = get_converter()
    all_mapped = []
    for sanction_data in sanction_docs[start_index:end_index]:
        name = sanction_data["entity"]["data"]["names"][0]["name"]
        source_mention_id = sanction_data["entity_id"]
        primary_ids = get_primary_ids(name)
        try:
            temp_df = get_source_mapping(
                source_mention_id, primary_ids, entity_version_converter, entity_store
            )
            all_mapped.append(temp_df)
        except Exception:
            logging.error(traceback.format_exc())

    df_all_mapped = pd.DataFrame(all_mapped)
    file_path = os.path.join(PATH, f"df_all_mapped_{str(start_index)}_{str(end_index)}.pickle")
    df_all_mapped.to_pickle(file_path)


if __name__ == "__main__":
    get_mapping()
