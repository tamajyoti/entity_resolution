import pandas as pd
import pytest

from am_combiner.combiners.common import (
    HOMOGENEITY_FIELD,
    COMPLETENESS_FIELD,
    V_SCORE_FIELD,
    COUNT_FIELD,
    NAME_OC_RATE_FIELD,
    NAME_UC_RATE_FIELD,
    PROFILES_PER_OC_FIELD,
    PROFILES_CREATED_FIELD,
    PROFILES_TRUE_FIELD,
    SCORE_TO_MINIMISE_FIELD,
    BLOCKING_FIELD_FIELD,
)
from am_combiner.features.article import Article
from am_combiner.qa.cross_validation import (
    random_draw,
    get_name_sensitivity_analysis,
    get_link_sensitivity_subsample,
)


def test_random_draw():
    assert len(random_draw([1, 2, 3, 4], holdout_ratio=0.5)) == 2


@pytest.mark.parametrize("holdout_ratio", (2.6, -1.8))
def test_random_draw_error(holdout_ratio):
    with pytest.raises(ValueError):
        random_draw([1, 2, 3], holdout_ratio=holdout_ratio)


def test_get_name_sensitivity_analysis():
    quality = pd.DataFrame(
        {
            HOMOGENEITY_FIELD: [0],
            COMPLETENESS_FIELD: [0],
            V_SCORE_FIELD: [0],
            COUNT_FIELD: [0],
            NAME_OC_RATE_FIELD: [0],
            NAME_UC_RATE_FIELD: [0],
            PROFILES_PER_OC_FIELD: [0],
            PROFILES_CREATED_FIELD: [0],
            PROFILES_TRUE_FIELD: [0],
            SCORE_TO_MINIMISE_FIELD: [0],
        }
    )
    analysis = get_name_sensitivity_analysis(quality, resamplings=10, holdout_ratio=0.2)
    assert len(analysis) == 10


@pytest.mark.parametrize("global_link_resampling", (True, False))
def test_get_link_sensitivity_subsample(
    entity_name, other_entity_name, validation, global_link_resampling
):
    article_1 = Article(
        entity_name=entity_name,
        url="url.1",
        article_text="Some text 1",
    )
    article_2 = Article(
        entity_name=entity_name,
        url="url.2",
        article_text="Some text 2",
    )
    article_3 = Article(
        entity_name=entity_name,
        url="url.3",
        article_text="Some text 3",
    )
    article_4 = Article(
        entity_name=entity_name,
        url="url.4",
        article_text="Some text 4",
    )
    article_5 = Article(
        entity_name=entity_name,
        url="url.5",
        article_text="Some text 5",
    )
    article_6 = Article(
        entity_name=entity_name,
        url="url.6",
        article_text="Some text 6",
    )
    article_7 = Article(
        entity_name=entity_name,
        url="url.7",
        article_text="Some text 7",
    )
    article_8 = Article(
        entity_name=other_entity_name,
        url="url.1",
        article_text="Some other text 1",
    )
    article_9 = Article(
        entity_name=other_entity_name,
        url="url.2",
        article_text="Some other text 2",
    )
    articles = {
        entity_name: [article_1, article_2, article_3, article_4, article_5, article_6, article_7],
        other_entity_name: [article_8, article_9],
    }
    article_sample, validation_sample = get_link_sensitivity_subsample(
        articles, validation, link_holdout_ratio=0.5, global_link_resampling=global_link_resampling
    )

    if global_link_resampling:
        article_sample = [
            article for name, all_articles in article_sample.items() for article in all_articles
        ]
        assert len(article_sample) == 5
        assert len(validation_sample) == 5
    else:
        assert len(article_sample[entity_name]) == 4
        assert len(article_sample[other_entity_name]) == 1

        assert len(validation_sample[validation_sample[BLOCKING_FIELD_FIELD] == entity_name]) == 4
        assert (
            len(validation_sample[validation_sample[BLOCKING_FIELD_FIELD] == other_entity_name])
            == 1
        )
