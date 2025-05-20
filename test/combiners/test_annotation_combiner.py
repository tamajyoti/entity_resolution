import numpy as np
from scipy import sparse

from am_combiner.combiners.annotation import AnnotationsCombiner
from am_combiner.features.article import Features, Article

from am_combiner.combiners.common import CLUSTER_NUMBER_FIELD


class FakeData:
    def __init__(self):

        self.combiner = AnnotationsCombiner([])

        article_1 = Article("Bob", "", "url1")
        article_2 = Article("Bob", "", "url2")
        article_3 = Article("Bob", "", "url3")

        tfidf_value = sparse.csr_matrix(np.array([1, 0, 0]))
        article_1.extracted_entities[Features.TFIDF_FULL_TEXT] = tfidf_value
        article_2.extracted_entities[Features.TFIDF_FULL_TEXT] = tfidf_value
        article_3.extracted_entities[Features.TFIDF_FULL_TEXT] = tfidf_value

        self.input_articles = [article_1, article_2, article_3]


fake_data = FakeData()


def test_negative_annotations_deleting_edges():

    fake_data.input_articles[0].positive_urls = []
    fake_data.input_articles[1].positive_urls = []
    fake_data.input_articles[2].positive_urls = []

    fake_data.input_articles[0].negative_urls = ["url2", "url3"]
    fake_data.input_articles[1].negative_urls = ["url1", "url3"]
    fake_data.input_articles[2].negative_urls = ["url1", "url2"]

    combiner_df = fake_data.combiner.combine_entities(fake_data.input_articles)

    assert combiner_df[CLUSTER_NUMBER_FIELD].tolist() == [0, 1, 2]


def test_positive_annotations_adding_edges():

    fake_data.input_articles[0].positive_urls = ["url2", "url3"]
    fake_data.input_articles[1].positive_urls = ["url1", "url3"]
    fake_data.input_articles[2].positive_urls = ["url1", "url2"]

    fake_data.input_articles[0].negative_urls = []
    fake_data.input_articles[1].negative_urls = []
    fake_data.input_articles[2].negative_urls = []

    combiner_df = fake_data.combiner.combine_entities(fake_data.input_articles)

    assert combiner_df[CLUSTER_NUMBER_FIELD].tolist() == [0, 0, 0]


def test_annotations_creating_two_clusters():

    fake_data.input_articles[0].positive_urls = ["url2"]
    fake_data.input_articles[1].positive_urls = ["url1"]
    fake_data.input_articles[2].positive_urls = []

    fake_data.input_articles[0].negative_urls = ["url3"]
    fake_data.input_articles[1].negative_urls = ["url3"]
    fake_data.input_articles[2].negative_urls = ["url1", "url2"]

    combiner_df = fake_data.combiner.combine_entities(fake_data.input_articles)
    assert combiner_df[CLUSTER_NUMBER_FIELD].tolist() == [0, 0, 1]
