import random
from typing import Iterable, Set, Dict, Tuple
from urllib.parse import urlparse
from random import shuffle
import pandas as pd
import numpy as np
from collections import defaultdict

from am_combiner.combiners.common import (
    UNIQUE_ID_FIELD,
    ENTITY_NAME_FIELD,
    TEXT_COLUMN_FIELD,
    GROUND_TRUTH_FIELD,
    CLUSTER_ID_GLOBAL_FIELD,
    BLOCKING_FIELD_FIELD,
    META_DATA_FIELD,
)
from am_combiner.utils.replace import replace_entity_name

ARTICLE_NUM = "article_num"
MAX_ATTEMPTS = 10000


def postprocess_fake_dataframe(input_df: pd.DataFrame) -> pd.DataFrame:
    """Rename fields, filter text and prepare the DF to be consumed by downstream."""
    df = input_df.copy()
    df.rename(
        columns={"entity_name": "original_entity_name", "pseudo_entity_name": BLOCKING_FIELD_FIELD},
        inplace=True,
    )

    df["content"] = df.apply(
        lambda row: replace_entity_name(
            row.content, row.original_entity_name, row[BLOCKING_FIELD_FIELD]
        ),
        axis=1,
    )
    df[CLUSTER_ID_GLOBAL_FIELD] = df["original_entity_name"].astype("category").cat.codes
    return df


def add_metadata_series(input_df: pd.DataFrame, meta_keys: Tuple[str, ...]):
    """Metadata handling."""
    if meta_keys:
        input_df[META_DATA_FIELD] = input_df[list(meta_keys)].to_dict(orient="records")
    return input_df


def preprocess_input_dataframe(
    input_df: pd.DataFrame, meta_keys: Tuple[str, ...] = ()
) -> pd.DataFrame:
    """Remove duplicates, replace links, etc."""
    default_columns = [ENTITY_NAME_FIELD, UNIQUE_ID_FIELD, TEXT_COLUMN_FIELD]
    default_columns.extend(meta_keys)
    input_df = input_df[default_columns]
    input_df = add_metadata_series(input_df, meta_keys)
    input_df[UNIQUE_ID_FIELD] = input_df[UNIQUE_ID_FIELD].apply(
        lambda url: urlparse(url)._replace(scheme="http").geturl()
    )
    input_df = input_df.drop_duplicates(subset=[UNIQUE_ID_FIELD], keep="first")
    # Make sure we only keep articles with the entity name inside it
    has_entity_mask = input_df.apply(
        lambda row: row[ENTITY_NAME_FIELD] in row[TEXT_COLUMN_FIELD], axis=1
    )
    input_df = input_df.loc[has_entity_mask].copy()
    input_df.reset_index(inplace=True, drop=True)
    return input_df


def get_articles_for_names(
    input_df: pd.DataFrame, names: Iterable[str]
) -> Tuple[Dict[str, Set[str]], Set[str]]:
    """Return per-name article set, and the global set of articles."""
    all_urls = set()
    names_urls: Dict[str, Set[str]] = {}
    for name in names:
        name_urls = set(input_df.loc[input_df[ENTITY_NAME_FIELD] == name, UNIQUE_ID_FIELD])
        names_urls[name] = name_urls
        all_urls.update(name_urls)

    return names_urls, all_urls


class SourceDataNotRichEnough(Exception):

    """Thrown if there is not enough data to generate the required fake data."""

    pass


class NoMoreValuesToPoll(Exception):

    """Custom exception to indicate we've run out of names to sample."""

    pass


class NameUrlSampler:

    """For a set of names performs sampling of corresponding urls at random."""

    def __init__(self, names_urls: Dict[str, Iterable[str]], name_weights="random"):
        if name_weights not in ("random", "equal"):
            raise ValueError("name_weights can either be `random` or `equal`")
        self.names_urls = names_urls
        # Ensure the values are lists, so we can shuffle them
        for k, v in self.names_urls.items():
            v_list = list(v)
            shuffle(v_list)
            self.names_urls[k] = v_list

        self.names = sorted(list(names_urls.keys()))
        if name_weights == "equal":
            self.weights = [1] * len(self.names)
        else:
            self.weights = [random.randint(1, 10) for _ in self.names]

    def pop_url_for_name(self, name: str) -> str:
        """For a given name, return pop a url from the set of corresponding urls."""
        if not self.names_urls.get(name):
            raise KeyError("Requested name has been exhausted or never existed.")
        this_url = self.names_urls[name].pop()
        if not self.names_urls[name]:
            ind = self.names.index(name)
            del self.names[ind]
            del self.weights[ind]
        return this_url

    def get_random_url(self) -> str:
        """Return an url for a random name."""
        if not self.names:
            raise NoMoreValuesToPoll("No more values left")
        this_time_name = random.choices(population=self.names, weights=self.weights)[0]
        return self.pop_url_for_name(this_time_name)


def get_random_data_set(
    name_set_distribution_summarizer,
    true_profiles_distribution_summarizer,
    article_data: pd.DataFrame,
    number_of_entities: int = 10,
    tag: str = "FAKE IDENTITY",
    meta_keys: Tuple[str, ...] = (),
) -> pd.DataFrame:
    """
    Generate anonymized data.

    Parameters
    ----------
    article_data:
        pd.DataFrame to sample the data from.
    true_profiles_distribution_summarizer:
        an object that returns the a histogram of number of true profiles and their sampling weights
    name_set_distribution_summarizer:
        an object that returns the a histogram of number of mentions to be generated.
    number_of_entities:
        The number of random sets that needs to be generated. In case the number is set to 5,
        it will generate 5 random sets of articles and entity names.
    tag:
        The tag to create pseudo entity name.
    meta_keys:
        The columns other than the default one to be selected
    Returns
    -------
        A dataframe containing the generated anonymized sample of data.

    Args:
        article_data:

    """
    article_data = preprocess_input_dataframe(article_data, meta_keys)

    names_to_artcile_num = article_data.groupby(by=ENTITY_NAME_FIELD)[[ENTITY_NAME_FIELD]].count()
    names_to_artcile_num.columns = [ARTICLE_NUM]
    names_to_artcile_num.sort_values(by=ARTICLE_NUM, ascending=False, inplace=True)

    true_profiles = true_profiles_distribution_summarizer.true_profiles
    weights = true_profiles_distribution_summarizer.weights

    unique_names = article_data[ENTITY_NAME_FIELD].unique().tolist()
    number_of_mentions_items, number_of_mention_weights = zip(
        *name_set_distribution_summarizer.number_of_mentions.items()
    )

    fake_batches = []

    def get_random_subset(src, wghts=None, size=1, replace=True):
        if wghts is None:
            p = None
        else:
            p = np.array(wghts) / sum(wghts)
        return np.random.choice(src, size=size, replace=replace, p=p)

    profile_nums = get_random_subset(true_profiles, weights, number_of_entities)
    article_nums = get_random_subset(
        number_of_mentions_items, number_of_mention_weights, number_of_entities
    )

    print(f"Have {article_data.shape[0]} articles, sampling: {sum(article_nums)}.")
    print(f"Have {len(unique_names)} unique names, sampling: {sum(profile_nums)}.")

    # Assumption: #true profiles num and #mention num distributions are highly correlated:
    profile_nums = sorted(profile_nums, reverse=True)
    article_nums = sorted(article_nums, reverse=True)

    name_usage = defaultdict(int)
    for profile_num, article_num in zip(profile_nums, article_nums):

        keep_searching, i = True, 0
        while (i < MAX_ATTEMPTS) and keep_searching:
            i += 1
            selected_names = get_random_subset(
                names_to_artcile_num.index, size=profile_num, replace=False
            )
            total_articles = names_to_artcile_num.loc[selected_names][ARTICLE_NUM].sum()
            if article_num <= total_articles:
                keep_searching = False

        if keep_searching:
            raise SourceDataNotRichEnough(
                "Do you have source data to sample from?"
                + f"Could not find {article_num} articles using {profile_num} true profiles"
            )

        for name in selected_names:
            name_usage[name] += 1
        names_url, all_urls = get_articles_for_names(article_data, selected_names)
        url_producer = NameUrlSampler(names_url, name_weights="random")
        url_list = []
        for ct in range(article_num):
            this_url = url_producer.get_random_url()
            url_list.append(this_url)
        fake_entity_batch = article_data[article_data[UNIQUE_ID_FIELD].isin(url_list)].copy()
        fake_entity_batch["pseudo_entity_name"] = f"{tag}{len(fake_batches)}"
        fake_entity_batch[GROUND_TRUTH_FIELD] = (
            fake_entity_batch[ENTITY_NAME_FIELD].astype("category").cat.codes
        )
        fake_batches.append(fake_entity_batch)

    random.shuffle(fake_batches)
    fake_entities_articles = pd.concat(fake_batches, ignore_index=True)
    fake_entities_articles_final = postprocess_fake_dataframe(fake_entities_articles)

    return fake_entities_articles_final
