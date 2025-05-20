from am_combiner.combiners.common import (
    CLUSTER_NUMBER_FIELD,
    UNIQUE_ID_FIELD,
    BLOCKING_FIELD_FIELD,
)
from am_combiner.combiners.graph_based import ConnectedComponentsCombiner
from am_combiner.features.article import Features, Article
from pandas.testing import assert_frame_equal
import pandas as pd


class TestConnectedComponentsCombiner:
    def test_params_are_properly_initialized(self):
        combiner = ConnectedComponentsCombiner(use_features=["a", Features.ORG, "B"], th=65)
        assert combiner.use_features == ["a", Features.ORG, "B"]
        assert combiner.th == 65

    def test_articles_with_one_features_in_common_are_combiner(self):
        a1 = Article(entity_name="A", article_text="A is a bad guy", url="http://a.com")
        a1.extracted_entities["A"] = {"A", "B", "C"}
        a2 = Article(entity_name="A", article_text="A is a very bad guy", url="http://aa.com")
        a2.extracted_entities["A"] = {"C", "D", "E"}

        combiner = ConnectedComponentsCombiner(use_features=["A"], th=1)
        result = combiner.combine_entities([a1, a2])

        expected = pd.DataFrame(
            {
                CLUSTER_NUMBER_FIELD: [0, 0],
                UNIQUE_ID_FIELD: ["http://a.com", "http://aa.com"],
                BLOCKING_FIELD_FIELD: ["A", "A"],
            }
        )
        assert_frame_equal(result, expected)

    def test_articles_with_nothing_in_common_are_not_combined(self):
        a1 = Article(entity_name="A", article_text="A is a bad guy", url="http://a.com")
        a1.extracted_entities["A"] = {"Aa", "Bb", "Cc"}
        a2 = Article(entity_name="A", article_text="A is a very bad guy", url="http://aa.com")
        a2.extracted_entities["A"] = {"C", "D", "E"}

        combiner = ConnectedComponentsCombiner(use_features=["A"], th=1)
        result = combiner.combine_entities([a1, a2])

        expected = pd.DataFrame(
            {
                CLUSTER_NUMBER_FIELD: [0, 1],
                UNIQUE_ID_FIELD: ["http://a.com", "http://aa.com"],
                BLOCKING_FIELD_FIELD: ["A", "A"],
            }
        )
        assert_frame_equal(result, expected)
