import abc
import json
import os
import random
from itertools import zip_longest
from typing import Dict, List, Union, Optional, Any

import pandas as pd
from pymongo import MongoClient

from am_combiner.features.article import Article
from am_combiner.articles import from_mongo_to_article_df
from am_combiner.config import read_mongo
from am_combiner.utils.random_data import get_random_data_set, add_metadata_series
from am_combiner.combiners.common import (
    URL_FIELD,
    TEXT_COLUMN_FIELD,
    CLUSTER_ID_FIELD,
    BLOCKING_FIELD_FIELD,
    CLUSTER_NUMBER_FIELD,
    META_DATA_FIELD,
    UNIQUE_ID_FIELD,
    ENTITY_NAME_FIELD,
    load_combiner_input_csv,
    combine_entities_wrapper,
    Combiner,
)


# Column names as appear in csv file:
CSV_ENTITY_NAME = "Entity Name"
CSV_URL_1 = "First Mention URL"
CSV_URL_2 = "Second Mention URL"
CSV_CONTENT_1 = "First Mention Article Text"
CSV_CONTENT_2 = "Second Mention Article Text"
CSV_ANNOTATION_RESULT = "Scale AI Answer"
CSV_POSITIVE_ANNOTATION = "Yes"
CSV_NEGATIVE_ANNOTATION = "No"

ANNOTATION_COMBINER = "AnnotationsCombiner"


def batch_iterable(source: List[Any], batch_size) -> List[List[Any]]:
    """Transform a list into a list of lists."""
    it = iter(source)
    output = [list(batch) for batch in zip_longest(*[it] * batch_size)]
    while output[-1][-1] is None:
        output[-1].pop()

    return output


def data_iterator_for_test_format(data_folder: str):
    """
    Return a generator with entity name and json file paths from DQS test data.

    Parameters
    ----------
    data_folder:
        A folder with all the names.

    Returns
    -------
        A generator object for all files in the folder.

    """
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Folder {data_folder} does not exist")
    for ct, name_dir in enumerate(os.listdir(data_folder)):
        join = os.path.join(data_folder, name_dir)
        if not os.path.isdir(join):
            continue
        article_files = os.listdir(join)
        for article_file in article_files:
            yield name_dir, os.path.join(data_folder, name_dir, article_file)


def json_iterator_for_test_format(data_folder: str):
    """
    Return a generator with entity name and json file content from DQS test data.

    Parameters
    ----------
    data_folder:
        A folder with all the names.

    Returns
    -------
        A generator over the json content in the test files.

    """
    for name_dir, json_file in data_iterator_for_test_format(data_folder):
        yield name_dir, json.load(open(json_file, "r"))


class AbstractInputDataProvider(abc.ABC):

    """
    An abstract class for obtaining combiner input dataframes from various sources.

    Generally, the user facing interface will obtain data and filter it according to the
    requested list of included/excluded names.

    It is up to the inheriting classes to implement strategy how a dataframe is obtained
    (e.g. read from a csv or mongodb instance, or some other custom filesystem-based structure)

    Attributes
    ----------
    params: Dict
        Specifies a dictionary of parameters that are required to instantiate any inheriting
        classes.
    entity_names: List[str]
        Limits which names will be returned to the user.
        If specified, then any entity name not included into this array will not be a part
        of the returned dataframe.
    excluded_entity_names: List[str]
        Specifies which names should be explicitly excluded from the dataframe.
        If overlaps with entity_names, then excluded_entity_names will over-rule whatever is
        specified in entity_names.
    max_names: Union[int, None]
        Limits the maximum number of names that will be returned to the user.
    min_content_length: int
        Limits the minimum length of a content that will be returned to the user.
    max_content_length: int
        Limits the maximum length of a content that will be returned to the user.

    """

    REQUIRED_ATTRIBUTES = []

    def __init__(
        self,
        params: Dict,
        entity_names: List[str] = None,
        excluded_entity_names: List[str] = None,
        max_names: Union[int, None] = None,
        min_content_length: Optional[int] = None,
        max_content_length: Optional[int] = None,
    ) -> None:
        self.max_names = max_names
        self.max_content_length = max_content_length
        self.min_content_length = min_content_length
        self.excluded_entity_names = excluded_entity_names
        self.entity_names = entity_names
        self.params = params
        self._validate_params()

    def _validate_params(self) -> None:
        """
        Check if all required parameters have been passed to the constructor.

        Raises
        ------
            ValueError if any parameters are missing

        """
        for param in self.REQUIRED_ATTRIBUTES:
            if param in self.params:
                continue
            raise ValueError(f"{param} is required to initialise a {self.__class__} instance")

    @abc.abstractmethod
    def _get_dataframe(self) -> pd.DataFrame:
        """
        Define how a dataframe is obtained.

        Returns
        -------
            A dataframe containing input data.

        """

    def get_dataframe(self) -> pd.DataFrame:
        """
        Obtain a dataframe.

        User-facing method, calls an abstract method and filters the results.

        Returns
        -------
            Filtered and ready to use dataframe which will serve as input.

        """
        entities_df = self._get_dataframe()
        if self.entity_names:
            entities_df = entities_df[entities_df[BLOCKING_FIELD_FIELD].isin(self.entity_names)]
        if self.excluded_entity_names:
            entities_df = entities_df[
                ~entities_df[BLOCKING_FIELD_FIELD].isin(self.excluded_entity_names)
            ]
        # Obtain the list of articles to loop through. Truncate with max_names if this is set
        unique_names = entities_df[BLOCKING_FIELD_FIELD].unique()
        if self.max_names:
            name_filter = set(unique_names[: self.max_names])
            entities_df = entities_df[entities_df[BLOCKING_FIELD_FIELD].isin(name_filter)]
        entities_df["content_length"] = entities_df.apply(lambda row: len(row.content), axis=1)
        if self.max_content_length:
            entities_df = entities_df[
                entities_df["content_length"] <= self.max_content_length
            ].reset_index(drop=True)
        if self.min_content_length:
            entities_df = entities_df[
                entities_df["content_length"] >= self.min_content_length
            ].reset_index(drop=True)
        return entities_df


class MongoDataProvider(AbstractInputDataProvider):

    """Implements a data provider which gets input data from Mongo."""

    REQUIRED_ATTRIBUTES = ["mongo_uri", "mongo_database", "mongo_collection", "meta_keys"]

    def __init__(
        self,
        params: Dict,
        entity_names: List[str] = None,
        excluded_entity_names: List[str] = None,
        max_names=None,
        min_content_length=None,
        max_content_length=None,
    ) -> None:
        super().__init__(
            params,
            entity_names,
            excluded_entity_names,
            max_names,
            min_content_length,
            max_content_length,
        )
        self.mongo_uri = self.params["mongo_uri"]
        self.mongo_database = self.params["mongo_database"]
        self.mongo_collection = self.params["mongo_collection"]
        self.meta_data = self.params["meta_keys"]

    def _get_dataframe(self) -> pd.DataFrame:
        """
        Get input data from Mongo.

        Returns
        -------
            A dataframe containing input data.

        """
        client = MongoClient(self.mongo_uri)
        entities_df = from_mongo_to_article_df(
            client, db=self.mongo_database, col=self.mongo_collection, meta_data=self.meta_data
        )
        entities_df = add_metadata_series(entities_df, self.meta_data)
        return entities_df


class RandomDataProvider(MongoDataProvider):

    """Implements a data provider which gets random input data from Mongo."""

    REQUIRED_ATTRIBUTES = [
        "random_input_size",
        "name_set_distribution_summarizer_class",
        "true_profiles_distribution_summarizer_class",
        "tag",
        "mongo_uri",
        "mongo_database",
        "mongo_collection",
        "meta_keys",
    ]

    def __init__(
        self,
        params: Dict,
        entity_names: List[str] = None,
        excluded_entity_names: List[str] = None,
        max_names=None,
        min_content_length=None,
        max_content_length=None,
    ) -> None:
        super().__init__(
            params,
            entity_names,
            excluded_entity_names,
            max_names,
            min_content_length,
            max_content_length,
        )
        self.random_input_size = self.params["random_input_size"]
        self.name_set_distribution_summarizer_class = self.params[
            "name_set_distribution_summarizer_class"
        ]
        self.profiles_distribution_summarizer_class = self.params[
            "true_profiles_distribution_summarizer_class"
        ]
        self.tag = self.params["tag"]
        self.meta_keys = self.params["meta_keys"]

    def _get_dataframe(self) -> pd.DataFrame:
        """
        Get random input data from Mongo.

        Returns
        -------
            A dataframe containing input data.

        """
        mongo_client = MongoClient(self.mongo_uri)
        dfs = []
        for mongo_collection in self.mongo_collection:
            print(f"Fetching collection {self.mongo_database}.{mongo_collection}")
            this_article_data = read_mongo(
                mongo_client, self.mongo_database, mongo_collection, {}, {}
            )
            dfs.append(this_article_data)

        article_data = pd.concat(dfs, ignore_index=True)
        article_data.rename(columns={"url": UNIQUE_ID_FIELD}, inplace=True)
        assign_random_dob_to_entities(article_data)
        return get_random_data_set(
            article_data=article_data,
            number_of_entities=self.random_input_size,
            name_set_distribution_summarizer=self.name_set_distribution_summarizer_class,
            true_profiles_distribution_summarizer=self.profiles_distribution_summarizer_class,
            tag=self.tag,
            meta_keys=self.meta_keys,
        )


def assign_random_dob_to_entities(
    input_df: pd.DataFrame, entity_name: str = ENTITY_NAME_FIELD, prob_dob: float = 0.1
):
    """
    Populate a dataframe with random DOBs.

    Args
    ----
        input_df: input dataframe to be populated
        entity_name: the name of the entity name column name
        prob_dob: the probability with which a dob needs to be assigned with

    Returns
    -------
        None

    """
    assert 0 <= prob_dob <= 1, f"prob_dob must be in range [0,1], given {prob_dob}"
    # First, generate true DOBs for original names
    dobs = {}
    for name in input_df[entity_name].unique():
        dobs[name] = random.randint(1900, 2020)

    # Generating DOBs for data itself

    def dob_generator(row):
        dob = dobs[row[entity_name]] if random.random() < prob_dob else None
        return dob

    input_df["DOB"] = input_df.apply(dob_generator, axis=1).astype(float)


class CSVDataProvider(AbstractInputDataProvider):

    """Implements a data provider which gets input data from a CSV file."""

    REQUIRED_ATTRIBUTES = ["input_csv"]

    def __init__(
        self,
        params: Dict,
        entity_names: List[str] = None,
        excluded_entity_names: List[str] = None,
        max_names=None,
        min_content_length=None,
        max_content_length=None,
    ) -> None:
        super().__init__(
            params,
            entity_names,
            excluded_entity_names,
            max_names,
            min_content_length,
            max_content_length,
        )
        self.input_csv = self.params["input_csv"]

    def _get_dataframe(self) -> pd.DataFrame:
        """
        Get input data from a CSV file.

        Returns
        -------
            A dataframe containing input data.

        """
        entities_df = load_combiner_input_csv(self.input_csv, ignore_missing_cols=True)
        return entities_df


class AnnotationsProvider(AbstractInputDataProvider):

    """Implements a data provider which gets pairwise annotation data."""

    REQUIRED_ATTRIBUTES = ["input_csv"]

    def __init__(
        self,
        params: Dict,
        entity_names: List[str] = None,
        excluded_entity_names: List[str] = None,
        max_names=None,
        min_content_length=None,
        max_content_length=None,
    ) -> None:
        super().__init__(
            params,
            entity_names,
            excluded_entity_names,
            max_names,
            min_content_length,
            max_content_length,
        )
        self.input_csv = self.params["input_csv"]
        self.annotation_df = pd.DataFrame([])

    def _add_listing_type(self, input_entities_df):
        """Lookup meta_data listing subtype."""
        col = MongoClient(self.params["mongo_uri"])[self.params["mongo_database"]][
            self.params["mongo_collection"]
        ]

        return_fields = {"url": 1, "entity_name": 1, "listing_subtype": 1, "_id": 0}
        articles_data = list(col.find({}, return_fields))
        article_df = pd.DataFrame(data=articles_data)

        input_entities_df = input_entities_df.merge(
            article_df,
            left_on=[BLOCKING_FIELD_FIELD, UNIQUE_ID_FIELD],
            right_on=["entity_name", "url"],
        )
        input_entities_df[META_DATA_FIELD] = input_entities_df["listing_subtype"].apply(
            lambda x: {"listing_subtype": x}
        )
        return input_entities_df

    def _get_dataframe(self) -> pd.DataFrame:
        """
        Get annotation pairs from a CSV file.

        Creates a dataframe containing all articles
        that has at least one annotation.

        Columns:
            - url
            - content
            - entity_name
        """
        self.annotation_df = pd.read_csv(self.input_csv)

        input_entities_df = pd.DataFrame(
            {
                UNIQUE_ID_FIELD: self.annotation_df[CSV_URL_1].tolist()
                + self.annotation_df[CSV_URL_2].tolist(),
                TEXT_COLUMN_FIELD: self.annotation_df[CSV_CONTENT_1].tolist()
                + self.annotation_df[CSV_CONTENT_2].tolist(),
                BLOCKING_FIELD_FIELD: self.annotation_df[CSV_ENTITY_NAME].tolist() * 2,
            }
        )

        input_entities_df.drop_duplicates(inplace=True)
        input_entities_df.sort_values(by=[BLOCKING_FIELD_FIELD], inplace=True)
        input_entities_df.reset_index(inplace=True, drop=True)
        input_entities_df = self._add_listing_type(input_entities_df)

        return input_entities_df

    def _store_pairwise_annotation_mappings(
        self, annotation_result: str
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        For given annotation result, wrangles annotation pairs into convenient data structure.

        Parameters
        ----------
        annotation_result:
            whether we are storing 'Yes' or 'No' annotation results
        Returns
        -------

            entity_name -> {url_1 -> all urls that have been paired
                            with url_1 for given entity and annotation result}

        """
        annotations_map = {}
        for entity_name in self.annotation_df[CSV_ENTITY_NAME].unique():
            annotations_map[entity_name] = {}

        for _, row in self.annotation_df.iterrows():

            # Only store the relevant answer type pairs:
            if row[CSV_ANNOTATION_RESULT] == annotation_result:

                # Store pair mapping twice (from both directions):
                for url1, url2 in [
                    (row[CSV_URL_1], row[CSV_URL_2]),
                    (row[CSV_URL_2], row[CSV_URL_1]),
                ]:
                    if url1 in annotations_map[row[CSV_ENTITY_NAME]]:
                        annotations_map[row[CSV_ENTITY_NAME]][url1].append(url2)
                    else:
                        annotations_map[row[CSV_ENTITY_NAME]][url1] = [url2]
        return annotations_map

    def _complement_articles_with_annotation_data(
        self, input_entities: Dict[str, List[Article]]
    ) -> None:
        """

        Complement articles with annotation data.

        Parameters
        ----------
        input_entities:
            dictionary of entity names mapping to list of articles.

        Returns
        -------
            Updates all articles with annotation information:
            article.positive_url = [urls that have positive annotation given entity_name and url]
            article.negative_url = [urls that have negative annotation given entity_name and url]

        """
        positive_annotations = self._store_pairwise_annotation_mappings(CSV_POSITIVE_ANNOTATION)
        negative_annotations = self._store_pairwise_annotation_mappings(CSV_NEGATIVE_ANNOTATION)

        for entity_name, entity_articles in input_entities.items():
            assert (
                entity_name in positive_annotations and entity_name in negative_annotations
            ), "Entity name is not found in annotation data"
            for article in entity_articles:

                if article.url in positive_annotations[entity_name]:
                    article.positive_urls = positive_annotations[entity_name][article.url]
                else:
                    article.positive_urls = []

                if article.url in negative_annotations[entity_name]:
                    article.negative_urls = negative_annotations[entity_name][article.url]
                else:
                    article.negative_urls = []

    @staticmethod
    def _add_cluster_ids_from_annotation_combiner(
        input_entities_df: pd.DataFrame, annotation_combiner_results_df: pd.DataFrame
    ) -> pd.DataFrame:
        """

        Merge annotation combiner clustering results as input_entities_df ground truth.

        Parameters
        ----------
        input_entities_df:
            data frame of [entity_name, url, context]
        annotation_combiner_results_df:
            data frame of annotation combiner result [entity_name, url, clustering ids]

        Returns
        -------
        Dataframe input_entities_df that now contains annotation
        combiner clustering results as ground truth.

        """
        original_article_num = input_entities_df.shape[0]

        if CLUSTER_ID_FIELD in input_entities_df.columns:
            del input_entities_df[CLUSTER_ID_FIELD]

        input_entities_df = input_entities_df.merge(
            annotation_combiner_results_df[
                [BLOCKING_FIELD_FIELD, UNIQUE_ID_FIELD, CLUSTER_NUMBER_FIELD]
            ],
            left_on=[BLOCKING_FIELD_FIELD, URL_FIELD],
            right_on=[BLOCKING_FIELD_FIELD, UNIQUE_ID_FIELD],
            how="inner",
        )
        input_entities_df.rename(columns={CLUSTER_NUMBER_FIELD: CLUSTER_ID_FIELD}, inplace=True)

        assert (
            input_entities_df.shape[0] == original_article_num
        ), "Combiner clustered articles do not match original input"
        return input_entities_df

    def get_ground_truth(
        self,
        input_entities_df: pd.DataFrame,
        input_entities: Dict[str, List[Article]],
        combiner_object: Combiner,
    ) -> pd.DataFrame:
        """

        Merge annotation combiner clustering results as input_entities_df ground truth.

        Parameters
        ----------
        input_entities_df:
            data frame of [entity_name, url, context]
        input_entities:
            articles objects
        combiner_object:
            annotation combiner that will produce ground truth clustering
        Returns
        -------
        Dataframe input_entities_df that now contains annotation
        combiner clustering results as ground truth.

        """
        self._complement_articles_with_annotation_data(input_entities)
        clustering_results_df, _ = combine_entities_wrapper(
            entity_articles=input_entities,
            combiner_object=combiner_object,
        )
        return self._add_cluster_ids_from_annotation_combiner(
            input_entities_df, clustering_results_df
        )


DATA_PROVIDERS_CLASS_MAPPING = {
    "MongoDataProvider": MongoDataProvider,
    "CSVDataProvider": CSVDataProvider,
    "RandomDataProvider": RandomDataProvider,
    "AnnotationsProvider": AnnotationsProvider,
}
