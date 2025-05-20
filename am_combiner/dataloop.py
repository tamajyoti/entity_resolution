import os
import pickle
from collections import defaultdict
from typing import Callable, List

import click
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from am_combiner.combiners.common import ENTITY_NAME_FIELD, URL_FIELD
from am_combiner.combiners.tfidf import identity_tokenizer
from am_combiner.features.article import Article
from am_combiner.features.article import Features
from am_combiner.features.frontend import ArticleFeatureExtractorFrontend
from am_combiner.features.mapping import VISITORS_CLASS_MAPPING
from am_combiner.utils.data import DATA_PROVIDERS_CLASS_MAPPING
from am_combiner.utils.parametrization import get_cache_from_yaml, features_str_to_enum


def tfidf_trainer(articles: List[Article]):
    """
    Train a tfidf vectoriser for a list of articles.

    Articles full texts are used as a corpus for training.

    Parameters
    ----------
    articles:
        A list of articles that are to be used as a corpus.

    """
    tfidf = TfidfVectorizer(
        min_df=5, max_df=0.95, max_features=8000, stop_words="english", ngram_range=(1, 3)
    )
    tfidf.fit([a.extracted_entities[Features.ARTICLE_TEXT] for a in articles])
    pickle.dump(tfidf, open("am_combiner/data/models/tfidf_1_3.pkl", "wb"))


def tfidf_coreference_resolved_trainer(articles: List[Article]):
    """
    Train a tfidf vectoriser for a list of articles.

    Articles full texts are used as a corpus for training.

    Parameters
    ----------
    articles:
        A list of articles that are to be used as a corpus.

    """
    tfidf = TfidfVectorizer(
        min_df=5, max_df=0.95, max_features=8000, stop_words="english", ngram_range=(1, 3)
    )
    tfidf.fit([a.extracted_entities[Features.COREFERENCE_RESOLVED_TEXT] for a in articles])
    pickle.dump(tfidf, open("am_combiner/data/models/tfidf_coreference_resolved.pkl", "wb"))


def tfidf_multi_vocab_trainer(articles: List[Article]):
    """
    Train a tfidf vectoriser for a list of articles.

    Parameters
    ----------
    articles:
        A list of articles that are to be used as a corpus.

    """
    with open("config.yaml") as f:
        tf_idf_vectoriser_pickle_filename_pattern = yaml.safe_load(f)[
            "TfIdfVectoriserPickleFilenamePattern"
        ]

    for vocab_size in tqdm(range(1000, 13000, 1000)):
        tfidf = TfidfVectorizer(
            min_df=5, max_df=0.95, max_features=vocab_size, stop_words="english", ngram_range=(1, 3)
        )
        tfidf.fit([a.extracted_entities[Features.ARTICLE_TEXT] for a in articles])
        this_fn = os.path.join(
            "am_combiner",
            "data",
            "models",
            tf_idf_vectoriser_pickle_filename_pattern.format(vocab_size=vocab_size),
        )
        pickle.dump(tfidf, open(this_fn, "wb"))


def tfidf_multi_freq_trainer(articles: List[Article]):
    """
    Train a tfidf vectoriser for a list of articles.

    Parameters
    ----------
    articles:
        A list of articles that are to be used as a corpus.

    """
    with open("config.yaml") as f:
        tf_idf_vectoriser_pickle_filename_pattern = yaml.safe_load(f)[
            "TfIdfVectoriserVariableFreqPickleFilenamePattern"
        ]

    visitors_strs = []
    for min_df in tqdm([1, 5, 20, 50]):
        for max_df in [0.95, 0.8, 0.6]:
            tfidf = TfidfVectorizer(
                min_df=min_df,
                max_df=max_df,
                max_features=8000,
                stop_words="english",
                ngram_range=(1, 3),
            )
            tfidf.fit([a.extracted_entities[Features.ARTICLE_TEXT] for a in articles])
            this_fn = os.path.join(
                "am_combiner",
                "data",
                "models",
                tf_idf_vectoriser_pickle_filename_pattern.format(min_df=min_df, max_df=max_df),
            )
            pickle.dump(tfidf, open(this_fn, "wb"))
            s = f"""- TFIDFFullTextVisitor_{min_df}_{max_df}:
      class: TFIDFFullTextVisitor
      attrs:
        vectoriser_pickle: {this_fn}"""
            visitors_strs.append(s)

    print("\n".join(visitors_strs))


def tfidf_trainer_no_names(articles: List[Article]):
    """
    Train a tfidf vectoriser for a list of articles.

    Articles full texts are used as a corpus for training.

    Parameters
    ----------
    articles:
        A list of articles that are to be used as a corpus.

    """
    tfidf = TfidfVectorizer(
        min_df=5, max_df=0.95, max_features=8000, stop_words="english", ngram_range=(1, 3)
    )
    tfidf.fit([a.extracted_entities[Features.ARTICLE_TEXT] for a in articles])
    pickle.dump(tfidf, open("am_combiner/data/models/tfidf_1_3_no_names.pkl", "wb"))


def tfidf_trainer_features(articles: List[Article]):
    """
    Train a tfidf vectoriser for a list of articles using features.

    Articles full texts are used as a corpus for training.

    Parameters
    ----------
    articles:
        A list of articles that are to be used as a corpus.

    """
    tfidf = TfidfVectorizer(
        min_df=5,
        max_df=0.95,
        max_features=8000,
        stop_words="english",
        ngram_range=(1, 3),
        tokenizer=identity_tokenizer,
    )

    tfidf.fit([a.extracted_entities[Features.FULL_TEXT_FEATURES] for a in articles])
    pickle.dump(tfidf, open("am_combiner/data/models/tfidf_features_1.pkl", "wb"))


class CallbackRegistry:

    """
    Class for handling callbacks.

    Can register new callbacks, get a certain callback by name.

    """

    CALLBACKS = {
        "tfidf_trainer": tfidf_trainer,
        "tfidf_coreference_resolved_trainer": tfidf_coreference_resolved_trainer,
        "tfidf_multi_vocab_trainer": tfidf_multi_vocab_trainer,
        "tfidf_trainer_no_names": tfidf_trainer_no_names,
        "tfidf_trainer_features": tfidf_trainer_features,
        "tfidf_multi_freq_trainer": tfidf_multi_freq_trainer,
    }

    @classmethod
    def register_new_callback(cls, callback_name: str, callback: Callable[[List[Article]], None]):
        """
        Allow one to add a new implementation if the main function is run outside of the script.

        Parameters
        ----------
        callback_name:
            A name for a callback.

        callback:
            A callable object that takes a list of Article objects and does something with them,
            can be feature extraction/validation/training/you name it.

        """
        cls.CALLBACKS.update({callback_name: callback})

    @classmethod
    def get_callback(cls, callback_name: str) -> Callable[[List[Article]], None]:
        """
        Return a callback for a registered function.

        Parameters
        ----------
        callback_name:
            A name for the callback to be returned.

        Returns
        -------
            Registered callback.

        """
        if callback_name not in cls.CALLBACKS:
            raise ValueError(
                f"Callback {callback_name} is not registered. Register it first before using."
            )
        return cls.CALLBACKS[callback_name]


@click.command()
@click.option(
    "--data-folder",
    type=click.Path(exists=True),
    help="Data folder with test set and names",
    default=None,
)
@click.option(
    "--callbacks",
    multiple=True,
    help="A set of callback to call",
    type=click.Choice(CallbackRegistry.CALLBACKS.keys()),
)
@click.option("--visitors", multiple=True, help="Define visitors to be run on the article")
@click.option(
    "--mongo-uri",
    help="Mongo URI for whatever reasons you may need it",
    default="mongodb://"
    "mongodb-mf-person-0-0.mi-playground-1"
    "/?replicaSet=mf-person-0&readPreference=primary&appname=MongoDB%20Compass&ssl=false",
)
@click.option(
    "--mongo-collection",
    help="The name of input collection to be used as a combiner input",
    default="dqs_validation_set",
)
@click.option(
    "--mongo-database",
    help="The name of input database to be used as a combiner input",
    default="am_combiner",
)
@click.option(
    "--max-names",
    type=int,
    default=None,
    help="Restrict the maximum number of names to be processed."
    "Use it if you need to process a subset of names",
)
def main(data_folder, callbacks, visitors, mongo_uri, mongo_collection, mongo_database, max_names):
    """Loop through and process input entities and cache them."""
    data_provider = "MongoDataProvider"
    provider = DATA_PROVIDERS_CLASS_MAPPING[data_provider](
        params={
            "mongo_uri": mongo_uri,
            "mongo_database": mongo_database,
            "mongo_collection": mongo_collection,
            "input_csv": None,
            "validation_df_for_distr": None,
            "random_input_size": None,
        },
        entity_names=None,
        excluded_entity_names=None,
        max_names=max_names,
    )

    input_entities_df = provider.get_dataframe()
    input_entities_df = input_entities_df.drop_duplicates(
        subset=[URL_FIELD, ENTITY_NAME_FIELD], keep="first"
    )
    visitors_cache = get_cache_from_yaml(
        "combiners_config.yaml",
        section_name="visitors",
        class_mapping=VISITORS_CLASS_MAPPING,
        restrict_classes=visitors,
        attrs_callbacks={
            "feature_name": features_str_to_enum,
            "source_feature": features_str_to_enum,
            "target_feature": features_str_to_enum,
        },
    )

    entity_articles = ArticleFeatureExtractorFrontend(
        visitors_cache=visitors_cache, visitors=visitors, thread_count=8
    ).produce_visited_objects_from_df(input_entities_df=input_entities_df)
    with open("am_combiner/data/cache/cache_random_50.pkl", "wb") as f:
        cache = defaultdict(dict)
        for entity_name, articles in entity_articles.items():
            for article in articles:
                cache[entity_name][article.url] = article
        pickle.dump(cache, f)


if __name__ == "__main__":
    main()
