import pytest
import pandas as pd
from am_combiner.utils.distributions import TrueProfilesDistribution, NameSetDistributionSummariser

from am_combiner.combiners.common import (
    CLUSTER_ID_GLOBAL_FIELD,
    TEXT_COLUMN_FIELD,
    ENTITY_NAME_FIELD,
    BLOCKING_FIELD_FIELD,
    UNIQUE_ID_FIELD,
    GROUND_TRUTH_FIELD,
)
from am_combiner.utils.random_data import (
    postprocess_fake_dataframe,
    preprocess_input_dataframe,
    get_articles_for_names,
    NameUrlSampler,
    NoMoreValuesToPoll,
    get_random_data_set,
    SourceDataNotRichEnough,
)

from pandas.testing import assert_frame_equal


class TestRandomData:
    @pytest.fixture
    def name_set_distribution_summarizer(self):
        return NameSetDistributionSummariser("test/utils/data/distributions/name_set_test.json")

    @pytest.fixture
    def name_set_distribution_summarizer_impossible(self):
        return NameSetDistributionSummariser(
            "test/utils/data/distributions/name_set_impossible_test.json"
        )

    @pytest.fixture
    def true_profiles_distribution_summarizer(self):
        return TrueProfilesDistribution(
            "test/utils/data/distributions/name_set_test_true_profiles.json"
        )

    @pytest.fixture
    def simple_article_data(self):
        return pd.DataFrame(
            {
                ENTITY_NAME_FIELD: ["name1", "name2", "name3"],
                TEXT_COLUMN_FIELD: ["content1 name1", "content2 name2", "content3 name3"],
                UNIQUE_ID_FIELD: ["http1", "http2", "http3"],
            }
        )

    @pytest.mark.parametrize(
        ["input_df", "expected_df"],
        [
            (
                pd.DataFrame(
                    {
                        "entity_name": ["name1", "name2"],
                        "pseudo_entity_name": ["fake1", "fake2"],
                        "content": ["name1 is good", "name2 bad"],
                    }
                ),
                pd.DataFrame(
                    {
                        "original_entity_name": ["name1", "name2"],
                        BLOCKING_FIELD_FIELD: ["fake1", "fake2"],
                        "content": ["fake1 is good", "fake2 bad"],
                        CLUSTER_ID_GLOBAL_FIELD: [0, 1],
                    }
                ),
            ),
            (
                pd.DataFrame(
                    {
                        "entity_name": ["name1", "name1"],
                        "pseudo_entity_name": ["fake1", "fake1"],
                        "content": ["name1 is good", "name1 bad"],
                    }
                ),
                pd.DataFrame(
                    {
                        "original_entity_name": ["name1", "name1"],
                        BLOCKING_FIELD_FIELD: ["fake1", "fake1"],
                        "content": ["fake1 is good", "fake1 bad"],
                        CLUSTER_ID_GLOBAL_FIELD: [0, 0],
                    }
                ),
            ),
        ],
    )
    def test_df_post_processing(self, input_df, expected_df):
        # A small ugly hack, since pandas will internally convert
        # category type to lesser integers, if not too many unique labels
        expected_df[CLUSTER_ID_GLOBAL_FIELD] = expected_df[CLUSTER_ID_GLOBAL_FIELD].astype("int8")
        assert_frame_equal(postprocess_fake_dataframe(input_df), expected_df)

    @pytest.mark.parametrize(
        ["input_df", "expected_df"],
        [
            (
                pd.DataFrame(
                    {
                        ENTITY_NAME_FIELD: ["name1", "name2", "name3"],
                        UNIQUE_ID_FIELD: [
                            "http://www.a.com",
                            "http://www.b.com",
                            "http://www.c.com",
                        ],
                        TEXT_COLUMN_FIELD: ["name1 a", "name2 b", "name3 c"],
                        "DONT TAKE THIS COLUMN": [1, 2, 3],
                    }
                ),
                pd.DataFrame(
                    {
                        ENTITY_NAME_FIELD: ["name1", "name2", "name3"],
                        UNIQUE_ID_FIELD: [
                            "http://www.a.com",
                            "http://www.b.com",
                            "http://www.c.com",
                        ],
                        TEXT_COLUMN_FIELD: ["name1 a", "name2 b", "name3 c"],
                    }
                ),
            )
        ],
    )
    def test_only_required_fields_are_taken(self, input_df, expected_df):
        assert_frame_equal(preprocess_input_dataframe(input_df), expected_df)

    @pytest.mark.parametrize(
        ["input_df", "expected_df"],
        [
            (
                pd.DataFrame(
                    {
                        ENTITY_NAME_FIELD: ["name1", "name2", "name3"],
                        UNIQUE_ID_FIELD: [
                            "http://www.a.com",
                            "https://www.b.com",
                            "https://www.c.com",
                        ],
                        TEXT_COLUMN_FIELD: ["name1 a", "name2 b", "name3 c"],
                        "DONT TAKE THIS COLUMN": [1, 2, 3],
                    }
                ),
                pd.DataFrame(
                    {
                        ENTITY_NAME_FIELD: ["name1", "name2", "name3"],
                        UNIQUE_ID_FIELD: [
                            "http://www.a.com",
                            "http://www.b.com",
                            "http://www.c.com",
                        ],
                        TEXT_COLUMN_FIELD: ["name1 a", "name2 b", "name3 c"],
                    }
                ),
            )
        ],
    )
    def test_http_schemas_are_replaced(self, input_df, expected_df):
        assert_frame_equal(preprocess_input_dataframe(input_df), expected_df)

    @pytest.mark.parametrize(
        ["input_df", "expected_df"],
        [
            (
                pd.DataFrame(
                    {
                        ENTITY_NAME_FIELD: ["name1", "name1", "name3"],
                        UNIQUE_ID_FIELD: [
                            "http://www.a.com",
                            "https://www.a.com",
                            "https://www.c.com",
                        ],
                        TEXT_COLUMN_FIELD: ["name1 a", "name1 b", "name3 c"],
                        "DONT TAKE THIS COLUMN": [1, 2, 3],
                    }
                ),
                pd.DataFrame(
                    {
                        ENTITY_NAME_FIELD: ["name1", "name3"],
                        UNIQUE_ID_FIELD: [
                            "http://www.a.com",
                            "http://www.c.com",
                        ],
                        TEXT_COLUMN_FIELD: ["name1 a", "name3 c"],
                    }
                ),
            ),
            (
                pd.DataFrame(
                    {
                        ENTITY_NAME_FIELD: ["name2", "name1", "name3"],
                        UNIQUE_ID_FIELD: [
                            "http://www.a.com",
                            "https://www.a.com",
                            "https://www.c.com",
                        ],
                        TEXT_COLUMN_FIELD: ["name2 a", "name1 b", "name3 c"],
                    }
                ),
                pd.DataFrame(
                    {
                        ENTITY_NAME_FIELD: ["name2", "name3"],
                        UNIQUE_ID_FIELD: [
                            "http://www.a.com",
                            "http://www.c.com",
                        ],
                        TEXT_COLUMN_FIELD: ["name2 a", "name3 c"],
                    }
                ),
            ),
        ],
    )
    def test_duplicates_are_removed(self, input_df, expected_df):
        assert_frame_equal(preprocess_input_dataframe(input_df), expected_df)

    @pytest.mark.parametrize(
        ["input_df", "expected_df"],
        [
            (
                pd.DataFrame(
                    {
                        ENTITY_NAME_FIELD: ["name1", "name1", "name3"],
                        UNIQUE_ID_FIELD: [
                            "http://www.a.com",
                            "https://www.b.com",
                            "https://www.c.com",
                        ],
                        TEXT_COLUMN_FIELD: ["name1 a", "b", "name3 c"],
                        "DONT TAKE THIS COLUMN": [1, 2, 3],
                    }
                ),
                pd.DataFrame(
                    {
                        ENTITY_NAME_FIELD: ["name1", "name3"],
                        UNIQUE_ID_FIELD: [
                            "http://www.a.com",
                            "http://www.c.com",
                        ],
                        TEXT_COLUMN_FIELD: ["name1 a", "name3 c"],
                    }
                ),
            )
        ],
    )
    def test_articles_wo_entity_name_are_not_taken(self, input_df, expected_df):
        assert_frame_equal(preprocess_input_dataframe(input_df), expected_df)

    @pytest.mark.parametrize(
        ["input_df", "names", "expected_dict", "expected_set"],
        [
            (
                pd.DataFrame(
                    {
                        ENTITY_NAME_FIELD: [
                            "name1",
                            "name1",
                            "name2",
                            "name3",
                            "name3",
                            "name4",
                        ],
                        UNIQUE_ID_FIELD: ["http1", "http2", "http3", "http5", "http6", "http6"],
                    }
                ),
                ["name1", "name2", "name3", "name4"],
                {
                    "name1": {"http1", "http2"},
                    "name2": {"http3"},
                    "name3": {"http5", "http6"},
                    "name4": {"http6"},
                },
                {"http1", "http2", "http3", "http5", "http6"},
            )
        ],
    )
    def test_urls_are_correcly_summarized(self, input_df, names, expected_dict, expected_set):
        names_url, all_urls = get_articles_for_names(input_df, names)
        assert names_url == expected_dict
        assert all_urls == expected_set

    @pytest.mark.parametrize(["name_weights"], [("A",), ("B",)])
    def test_can_not_be_created_with_wrong_args(self, name_weights):
        with pytest.raises(ValueError):
            NameUrlSampler({}, name_weights=name_weights)

    @pytest.mark.parametrize(["name_weights"], [("random",), ("equal",)])
    def test_can_be_created_with_valid_args(self, name_weights):
        NameUrlSampler({}, name_weights=name_weights)

    @pytest.mark.parametrize(["name_weights"], [("random",), ("equal",)])
    def test_weights_have_correct_length(self, name_weights):
        s = NameUrlSampler({"name1": set(), "name2": set()}, name_weights=name_weights)
        assert len(s.weights) == 2

    @pytest.mark.parametrize(
        ["names_urls"],
        [
            ({"B": {1, 2, 3}},),
            ({"A": set()},),
        ],
    )
    def test_raises_error_on_exhausted_name(self, names_urls):
        s = NameUrlSampler(names_urls)
        with pytest.raises(KeyError):
            s.pop_url_for_name("A")

    def test_pops_one_and_reports_exhausted(self):
        s = NameUrlSampler(names_urls={"A": {"http"}})
        url = s.pop_url_for_name("A")
        assert url == "http"
        with pytest.raises(KeyError):
            s.pop_url_for_name("A")
        assert s.names == []
        assert s.weights == []

    def test_internal_arrays_are_maintained_correctly(self):
        s = NameUrlSampler(names_urls={"A": {"http1"}, "B": {"http2"}})
        s.weights = [1, 2]
        s.names = ["A", "B"]
        url = s.pop_url_for_name("A")
        with pytest.raises(KeyError):
            s.pop_url_for_name("A")
        assert url == "http1"
        assert s.weights == [2]
        assert s.names == ["B"]

        url = s.pop_url_for_name("B")
        with pytest.raises(KeyError):
            s.pop_url_for_name("B")
        assert url == "http2"
        assert s.weights == []
        assert s.names == []

    def test_random_polling_properly_exhausts_samples(self):
        s = NameUrlSampler(names_urls={"A": {"http1", "http3", "http5"}, "B": {"http2", "http7"}})

        L = [s.get_random_url() for _ in range(5)]
        assert sorted(L) == ["http1", "http2", "http3", "http5", "http7"]

        with pytest.raises(NoMoreValuesToPoll):
            s.get_random_url()

        assert s.weights == []
        assert s.names == []

    def test_output_random_frame_has_required_columns(
        self,
        simple_article_data,
        name_set_distribution_summarizer,
        true_profiles_distribution_summarizer,
    ):
        random_data = get_random_data_set(
            name_set_distribution_summarizer,
            true_profiles_distribution_summarizer,
            article_data=simple_article_data,
            number_of_entities=1,
            tag="A",
        )
        assert "original_entity_name" in random_data.columns
        assert UNIQUE_ID_FIELD in random_data.columns
        assert CLUSTER_ID_GLOBAL_FIELD in random_data.columns
        assert GROUND_TRUTH_FIELD in random_data.columns

    @pytest.mark.parametrize(["number_of_entities"], [(1,), (34,)])
    @pytest.mark.parametrize(
        ["simple_article_data"],
        [
            (
                pd.DataFrame(
                    {
                        ENTITY_NAME_FIELD: ["name1", "name2", "name3"],
                        TEXT_COLUMN_FIELD: ["name1 content1", "name2 content2", "name3 content3"],
                        UNIQUE_ID_FIELD: ["http://a.com", "http://b.com", "http://c.com"],
                    }
                ),
            )
        ],
    )
    @pytest.mark.parametrize(
        ["name_set_distribution_summarizer", "true_profiles_distribution_summarizer"],
        [
            (
                NameSetDistributionSummariser("test/utils/data/distributions/name_set_test.json"),
                TrueProfilesDistribution(
                    "test/utils/data/distributions/name_set_test_true_profiles.json"
                ),
            )
        ],
    )
    def test_output_has_data_in_correct_format(
        self,
        number_of_entities,
        simple_article_data,
        name_set_distribution_summarizer,
        true_profiles_distribution_summarizer,
    ):
        random_data = get_random_data_set(
            name_set_distribution_summarizer,
            true_profiles_distribution_summarizer,
            article_data=simple_article_data,
            number_of_entities=number_of_entities,
            tag="NEWFAKETAG",
        )
        assert len(random_data[BLOCKING_FIELD_FIELD].unique()) == number_of_entities
        assert any(random_data["original_entity_name"].isin(simple_article_data[ENTITY_NAME_FIELD]))
        assert any(random_data[UNIQUE_ID_FIELD].isin(simple_article_data[UNIQUE_ID_FIELD]))
        assert all(random_data[BLOCKING_FIELD_FIELD].apply(lambda x: x.startswith("NEWFAKETAG")))

    def test_raises_error_with_impossible_requirements(
        self,
        simple_article_data,
        name_set_distribution_summarizer_impossible,
        true_profiles_distribution_summarizer,
    ):
        with pytest.raises(SourceDataNotRichEnough):
            get_random_data_set(
                name_set_distribution_summarizer_impossible,
                true_profiles_distribution_summarizer,
                article_data=simple_article_data,
                number_of_entities=1,
                tag="NEWFAKETAG",
            )
