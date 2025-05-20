import abc
import multiprocessing
import pickle
import time
from collections import defaultdict
from typing import List, Dict, Tuple, Iterable
import pandas as pd
from tqdm import tqdm
from pymongo import MongoClient
from io import BytesIO
from am_combiner.combiners.common import (
    TEXT_COLUMN_FIELD,
    META_DATA_FIELD,
    UNIQUE_ID_FIELD,
    SANCTION_ENTITY_FIELD,
    BLOCKING_FIELD_FIELD,
)
from am_combiner.features.article import Article
from am_combiner.features.sanction import Sanction
from am_combiner.features.common import ArticleVisitor
from am_combiner.utils.data import batch_iterable


def _process_article(params) -> Article:
    """
    Create an article object and apply visitors.

    Parameters
    ----------
    params:
        A tuple of article text, url, entity name and visitors to apply.

    Returns
    -------
        An Article object on which the list of visitors was applied.

    """
    (text, url, entity_name, meta_data), visitors = params
    article = Article(entity_name, text, url, meta_data)

    for visitor in visitors:
        article.accept_visitor(visitor)
    return article


def _process_sanction(params) -> Sanction:
    """Create a sanction object and apply visitors."""
    (sanction_id, raw_entity, sanction_type), visitors = params
    sanction = Sanction(sanction_id, raw_entity, sanction_type)

    for visitor in visitors:
        sanction.accept_visitor(visitor)
    return sanction


def _generate_visited_objects(
    input_tuples: List[Tuple],
    visitors: List[ArticleVisitor],
    thread_count: int = 4,
    processing_function=_process_article,
) -> Iterable[Article]:
    """
    Generate visited articles.

    Parameters
    ----------
    input_tuples:
        List of tuple inputs required to create desired object.
    visitors:
        A list of visitor objects.
    thread_count:
        Number of threads to carry out the processing in parallel.
    processing_function:
        function that creates desired objects and runs visitors

    Yields
    ------
        Visited articles

    """
    input_generator = ((input_tuple, visitors) for input_tuple in input_tuples)
    with multiprocessing.pool.ThreadPool(thread_count) as pool:
        yield from pool.imap(processing_function, input_generator, chunksize=4)


class AbstractFeatureExtractionFrontend(abc.ABC):

    """Abstract class which serves as a frontend for the feature extraction process."""

    @abc.abstractmethod
    def produce_visited_objects_from_df(
        self, input_entities_df: pd.DataFrame
    ) -> Dict[str, List[Article]]:
        """
        Create and visit articles and return them to the user, using the class configuration.

        Parameters
        ----------
        input_entities_df:
            The input dataframe that contains sufficient information for producing a bunch of
            visited articles.

        Returns
        -------
            A mapping of type entity_name -> List of visited articles.

        """


class ArticleFeatureExtractorFrontend(AbstractFeatureExtractionFrontend):

    """
    A class which serves as a frontend for the feature extraction process.

    Attributes
    ----------
    visitors_cache: Dict[str, ArticleVisitor]
        A dictionary containing a bunch of visitors that can be looked up by their name.
    visitors: List[str]
        A list of names that should be used to visit articles.
    thread_count: int
        Number of threads to use during the parallel execution.

    """

    def __init__(
        self,
        visitors_cache: Dict[str, ArticleVisitor],
        visitors: List[str] = None,
        thread_count: int = 4,
    ) -> None:
        self.visitors_cache = visitors_cache
        self.visitors = visitors if visitors else []
        for visitor in visitors:
            if visitor not in self.visitors_cache:
                raise ValueError(f"Visitor {visitor} does not exist in the given cache")
        self.thread_count = thread_count

    def produce_visited_objects_from_df(
        self, input_entities_df: pd.DataFrame
    ) -> Dict[str, List[Article]]:
        """
        Create and visit articles and return them to the user, using the class configuration.

        Parameters
        ----------
        input_entities_df:
            The input dataframe that contains sufficient information for producing a bunch of
            visited articles.

        Returns
        -------
            A mapping of type entity_name -> List of visited articles.

        """
        if META_DATA_FIELD in input_entities_df.columns:
            gen_meta_list = input_entities_df[META_DATA_FIELD].tolist()
        else:
            gen_meta_list = [{}] * input_entities_df.shape[0]

        article_inputs = zip(
            input_entities_df[TEXT_COLUMN_FIELD].tolist(),
            input_entities_df[UNIQUE_ID_FIELD].tolist(),
            input_entities_df[BLOCKING_FIELD_FIELD].tolist(),
            gen_meta_list,
        )
        visitors_objects = [self.visitors_cache[v] for v in self.visitors]
        args = (
            article_inputs,
            visitors_objects,
            self.thread_count,
            _process_article,
        )
        entity_articles = defaultdict(list)
        for article in tqdm(_generate_visited_objects(*args), total=input_entities_df.shape[0]):
            entity_articles[article.entity_name].append(article)
        return entity_articles


class SanctionFeatureExtractorFrontend(ArticleFeatureExtractorFrontend):

    """
    A class which serves as a frontend for the feature extraction process.

    Attributes
    ----------
    visitors_cache: Dict[str, ArticleVisitor]
        A dictionary containing a bunch of visitors that can be looked up by their name.
    visitors: List[str]
        A list of names that should be used to visit articles.
    thread_count: int
        Number of threads to use during the parallel execution.

    """

    def __init__(
        self,
        visitors_cache: Dict[str, ArticleVisitor],
        visitors: List[str] = None,
        thread_count: int = 4,
    ) -> None:
        super().__init__(visitors_cache, visitors, thread_count)

    def produce_visited_objects_from_df(
        self, input_entities_df: pd.DataFrame
    ) -> Dict[str, List[Sanction]]:
        """
        Create and visit articles and return them to the user, using the class configuration.

        Parameters
        ----------
        input_entities_df:
            The input dataframe that contains sufficient information for producing a bunch of
            visited sanctions.

        Returns
        -------
            A mapping of type entity_name -> List of visited sanctions.

        """
        total_sanctions = input_entities_df.shape[0]
        sanction_inputs = zip(
            input_entities_df[UNIQUE_ID_FIELD].tolist(),
            input_entities_df[SANCTION_ENTITY_FIELD].tolist(),
            input_entities_df[BLOCKING_FIELD_FIELD].tolist(),
        )

        visitors_objects = [self.visitors_cache[v] for v in self.visitors]
        args = (
            sanction_inputs,
            visitors_objects,
            self.thread_count,
            _process_sanction,
        )
        sanctions = defaultdict(list)
        for sanction in tqdm(_generate_visited_objects(*args), total=total_sanctions):
            sanctions[sanction.type].append(sanction)
        return sanctions


class FromCacheFeatureExtractionFrontend(AbstractFeatureExtractionFrontend):

    """
    A frontend reading features from the cache instead of calculating them.

    The frontend assumes that all required articles were already pre-processed and their
    features are stored in a given cache.

    """

    def __init__(
        self,
        mongo_uri: str,
        cache_database: str,
        cache_collection: str,
        take_name_key_from: str = "original_entity_name",
    ) -> None:
        self.cache_collection = cache_collection
        self.cache_database = cache_database
        self.mongo_uri = mongo_uri
        self.take_name_key_from = take_name_key_from

    def __fetch_feature_batch(self, urls):
        coll = MongoClient(self.mongo_uri)[self.cache_database][self.cache_collection]
        cursor = coll.find({"url": {"$in": urls}})
        article_dict = {}
        for doc in tqdm(cursor):
            article_dict[doc["url"]] = doc["extracted_features"]
        return article_dict

    def produce_visited_objects_from_df(
        self, input_entities_df: pd.DataFrame
    ) -> Dict[str, List[Article]]:
        """
        Create and visit articles and return them to the user, using the class configuration.

        Parameters
        ----------
        input_entities_df:
            The input dataframe that contains sufficient information for producing a bunch of
            visited articles.

        Returns
        -------
            A mapping of type entity_name -> List of visited articles.

        """
        now = time.time()
        urls_list = input_entities_df[UNIQUE_ID_FIELD].to_list()
        urls_batches = batch_iterable(urls_list, 5000)
        article_dict = {}
        for idx, url_batch in enumerate(urls_batches):
            print(f"Loading cache batch {idx+1} of {len(urls_batches)}")
            batch_dict = self.__fetch_feature_batch(url_batch)
            article_dict = {**article_dict, **batch_dict}
        print(f"cache reading elapsed {time.time() - now}")
        # To maintain the structure expected by combiners
        entity_articles = defaultdict(list)
        if META_DATA_FIELD not in input_entities_df:
            input_entities_df[META_DATA_FIELD] = [{}] * len(input_entities_df)
        for ct, row in input_entities_df.iterrows():
            article = Article(
                entity_name=row[BLOCKING_FIELD_FIELD],
                article_text=row.content,
                url=row[UNIQUE_ID_FIELD],
                meta=row.meta_data,
            )
            more_features = pickle.load(BytesIO(article_dict[row[UNIQUE_ID_FIELD]]))
            article.extracted_entities = defaultdict(
                set, {**article.extracted_entities, **more_features}
            )
            entity_articles[article.entity_name].append(article)

        return entity_articles
