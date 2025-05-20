import abc
import os
from pathlib import Path
from typing import Dict, List

import boto3
import numpy as np
import pandas as pd
from matplotlib import pyplot
from pymongo import MongoClient

from am_combiner.features.article import Article


class OutputUriKeeper:

    """Object to keep record on the output paths."""

    def __init__(self, output_path=None, dataframe_uri=None, results_dataframe_uri=None):
        self.output_path = output_path
        self._dataframe_uri = dataframe_uri
        self._results_dataframe_uri = results_dataframe_uri

    @property
    def dataframe_uri(self):
        """Get input uri."""
        return Path(self.output_path) / self._dataframe_uri

    @property
    def results_dataframe_uri(self):
        """Get the results uri."""
        return Path(self.output_path) / self._results_dataframe_uri


def ensure_s3_resource_exists(uri: str, target_folder: str):
    """Download an s3 resource if does not exist in the target_folder."""
    s3 = boto3.client("s3")
    p_uri = Path(uri)
    # S3 downloader will not create the path automatically
    Path(target_folder).mkdir(exist_ok=True)
    resource_fn = Path(target_folder) / p_uri.name
    if not os.path.exists(resource_fn):
        print(f"Downloading file from {uri}")
        s3.download_file(
            p_uri.parts[-2],  # bucket name that is
            p_uri.name,
            str(resource_fn),  # Converting posix path obj to str, to comply with interface
        )
        print("Done downloading")
    else:
        print(f"{p_uri.name} already exists in {target_folder}")

    return resource_fn


class AbstractResultsSaver(abc.ABC):

    """

    An interface each data saver must implement in order to store output data in various locations.

    Methods
    -------
    store_dataframe:
        Implements logic how to store dataframes on various resources.
    store_histogram_input:
         Implements logic how to store original histogram data on various resources.


    """

    REQUIRED_ATTRIBUTES = []

    def __init__(self, **kwargs):
        for a in self.REQUIRED_ATTRIBUTES:
            assert a in kwargs

    @abc.abstractmethod
    def store_dataframe(self, df: pd.DataFrame, uri: str) -> None:
        """Take a dataframe a store in a target location."""

    @abc.abstractmethod
    def store_histogram_input(self, histogram_input, uri: str) -> None:
        """Store a histogram in a target location."""


class LocalResultsSaver(AbstractResultsSaver):

    """

    IO wrapper that stores output data on a local HD.

    Attributes
    ----------
    output_path:
        An output path where to store output files.

    """

    REQUIRED_ATTRIBUTES = ["output_path"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_path = Path(kwargs["output_path"])
        self.output_path.mkdir(parents=True, exist_ok=True)

    def store_dataframe(self, df: pd.DataFrame, uri: str) -> None:
        """Concrete implementation of the abstract method."""
        df.to_csv(self.output_path / uri)

    def store_histogram_input(self, histogram_input: Dict, uri: str) -> None:
        """Concrete implementation of the abstract method."""
        assert "title" in histogram_input
        assert "data" in histogram_input
        pyplot.figure()
        pyplot.title(histogram_input["title"])
        pyplot.hist(histogram_input["data"])
        pyplot.savefig(self.output_path / uri)
        pyplot.close()


class MongoResultsSaver(AbstractResultsSaver):

    """

    IO wrapper that stores data in a mongo instance.

    Attributes
    ----------
    mongo_uri:
        URI for a mongo database
    database:
        database name used for results storage
    collection:
        collection name for results storage

    """

    REQUIRED_ATTRIBUTES = ["mongo_uri", "database", "collection"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mongo_uri = kwargs["mongo_uri"]
        self.database = kwargs["database"]
        self.collection = kwargs["collection"]

    def store_dataframe(self, df: pd.DataFrame, uri: str) -> None:
        """Concrete implementation of the abstract method."""
        client = MongoClient(self.mongo_uri)
        collection = client[self.database][f"{self.collection}-csv"]
        collection.insert({"uri": uri, "content": df.to_json()})

    def store_histogram_input(self, histogram_input, uri: str) -> None:
        """Concrete implementation of the abstract method."""
        # This method is left empty intentionally
        pass


STORAGE_MAPPING = {"local": LocalResultsSaver, "mongo": MongoResultsSaver}


def store_similarities(
    sim: np.ndarray,
    input_entities: List[Article],
    cluster_ids: List[int],
    mongo_client: MongoClient,
):
    """
    Store pairwise similarities of the input entities in a Mongo collection.

    Parameters
    ----------
    sim:
        Pairwise similarities of the input entities to store.
    input_entities:
        The input entities to store similarities for.
    cluster_ids:
        The clustering results, which will also be stored along with the similarities.
    mongo_client:
        The Mongo client, where to store the similarities.

    """
    similarity_documents = list()
    for first_index in range(len(sim)):
        for second_index in range(first_index + 1, len(sim)):
            similarity_documents.append(
                {
                    "Entity Name": input_entities[first_index].entity_name,
                    "First URL": input_entities[first_index].url,
                    "First Cluster ID": cluster_ids[first_index],
                    "Second URL": input_entities[second_index].url,
                    "Second Cluster ID": cluster_ids[second_index],
                    "Similarity": float(sim[first_index][second_index]),
                }
            )

    if similarity_documents:
        mongo_client.insert_many(documents=similarity_documents)
