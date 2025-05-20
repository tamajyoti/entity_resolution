import pandas as pd

from am_combiner.features.article import Article
from am_combiner.utils.data import AnnotationsProvider

from am_combiner.utils.data import (
    CSV_ENTITY_NAME,
    CSV_URL_1,
    CSV_URL_2,
    CSV_ANNOTATION_RESULT,
    CSV_POSITIVE_ANNOTATION,
    CSV_NEGATIVE_ANNOTATION,
)


def test_article_augmentation_with_positive_url():

    provider = AnnotationsProvider(params={"input_csv": ""})
    provider.annotation_df = pd.DataFrame(
        data={
            CSV_ENTITY_NAME: ["John"],
            CSV_URL_1: ["url1"],
            CSV_URL_2: ["url2"],
            CSV_ANNOTATION_RESULT: [CSV_POSITIVE_ANNOTATION],
        }
    )

    article_1 = Article("John", "", "url1")
    article_2 = Article("John", "", "url2")

    provider._complement_articles_with_annotation_data({"John": [article_1, article_2]})
    assert article_1.positive_urls == ["url2"]
    assert article_2.positive_urls == ["url1"]
    assert article_1.negative_urls == []
    assert article_2.negative_urls == []


def test_article_augmentation_with_negative_url():

    provider = AnnotationsProvider(params={"input_csv": ""})
    provider.annotation_df = pd.DataFrame(
        data={
            CSV_ENTITY_NAME: ["John"],
            CSV_URL_1: ["url1"],
            CSV_URL_2: ["url2"],
            CSV_ANNOTATION_RESULT: [CSV_NEGATIVE_ANNOTATION],
        }
    )

    article_1 = Article("John", "", "url1")
    article_2 = Article("John", "", "url2")

    provider._complement_articles_with_annotation_data({"John": [article_1, article_2]})
    assert article_1.negative_urls == ["url2"]
    assert article_2.negative_urls == ["url1"]
    assert article_1.positive_urls == []
    assert article_2.positive_urls == []
