import pytest
import numpy as np

from am_combiner.combiners.common import CLUSTER_ID_FIELD
from am_combiner.utils.data import AbstractInputDataProvider, assign_random_dob_to_entities
from am_combiner.utils.data import CSVDataProvider
from am_combiner.utils.data import MongoDataProvider
from am_combiner.utils.data import RandomDataProvider


class TestDataProviders:
    def test_assign_random_dob_to_entities(self, test_dataframe):
        assign_random_dob_to_entities(test_dataframe)
        assert "DOB" in test_dataframe.columns

    def test_assign_random_dob_to_entities_no_dobs_with_zero_prob(self, test_dataframe):
        assign_random_dob_to_entities(test_dataframe, prob_dob=0.0)
        for dob in test_dataframe["DOB"]:
            assert np.isnan(dob)

    def test_assign_random_dob_to_entities_all_dobs_with_one_prob(self, test_dataframe):
        assign_random_dob_to_entities(test_dataframe, prob_dob=1.0)
        for dob in test_dataframe["DOB"]:
            assert not np.isnan(dob)

    def test_abstract_provider_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            AbstractInputDataProvider(params={})

    def test_random_data_provider_requires_only_whats_needed(self):
        assert RandomDataProvider.REQUIRED_ATTRIBUTES == [
            "random_input_size",
            "name_set_distribution_summarizer_class",
            "true_profiles_distribution_summarizer_class",
            "tag",
            "mongo_uri",
            "mongo_database",
            "mongo_collection",
            "meta_keys",
        ]

    @pytest.mark.parametrize(
        ["params"],
        [
            ({"validation_df_for_distr": "mongodb://"},),
            ({"random_input_size": "mongodb://"},),
        ],
    )
    def test_random_data_provider_can_not_be_initialised_wo_all_required_params(self, params):
        with pytest.raises(ValueError):
            RandomDataProvider(params=params)

    def test_mongo_provider_requires_only_whats_needed(self):
        assert MongoDataProvider.REQUIRED_ATTRIBUTES == [
            "mongo_uri",
            "mongo_database",
            "mongo_collection",
            "meta_keys",
        ]

    @pytest.mark.parametrize(
        ["params"],
        [
            ({"mongo_uri": "mongodb://", "mongo_database": "am_combiner"},),
            ({"mongo_uri": "mongodb://", "mongo_collection": "collection"},),
            ({"mongo_database": "am_combiner", "mongo_collection": "collection"},),
        ],
    )
    def test_mongo_data_provider_can_not_be_initialised_wo_all_required_params(self, params):
        with pytest.raises(ValueError):
            MongoDataProvider(params=params)

    def test_mongo_data_provider_can_be_initialised(self):
        obj = MongoDataProvider(
            params={
                "mongo_uri": "mongodb://",
                "mongo_database": "am_combiner",
                "mongo_collection": "collection",
                "meta_keys": ("country",),
            }
        )
        assert obj.mongo_uri == "mongodb://"
        assert obj.mongo_database == "am_combiner"
        assert obj.mongo_collection == "collection"
        assert obj.meta_data == ("country",)

    def test_csv_provider_requires_only_whats_needed(self):
        assert CSVDataProvider.REQUIRED_ATTRIBUTES == ["input_csv"]

    def test_csv_data_provider_can_be_initialised(self, entity_names, excluded_entity_names):
        obj = CSVDataProvider(
            params={
                "input_csv": "//",
            },
            entity_names=entity_names,
            excluded_entity_names=excluded_entity_names,
        )
        assert obj.excluded_entity_names == excluded_entity_names
        assert obj.entity_names == entity_names

    def test_csv_data_provider_can_not_be_initialised_wo_input_csv(self):
        with pytest.raises(ValueError):
            CSVDataProvider(params={})

    def test_csv_data_provider_reads_and_filters_correctly(
        self, test_dataframe_as_csv_content, entity_names, excluded_entity_names
    ):
        obj = CSVDataProvider(
            params={
                "input_csv": test_dataframe_as_csv_content,
            },
            entity_names=entity_names,
            excluded_entity_names=excluded_entity_names,
        )
        df = obj.get_dataframe()
        assert df["blocking_field"].tolist() == ["A", "B", "C"]

    def test_a_data_provider_returns_proper_number_of_names_when_limited(
        self, test_dataframe_as_csv_content, entity_names, excluded_entity_names
    ):
        obj = CSVDataProvider(
            params={
                "input_csv": test_dataframe_as_csv_content,
            },
            entity_names=entity_names,
            excluded_entity_names=excluded_entity_names,
            max_names=2,
        )
        df = obj.get_dataframe()
        assert len(df["blocking_field"].tolist()) == 2


class TestDataUtils:
    def test_dataset_summariser_correctly_counts_clusters(self, dataset_summariser):
        assert dataset_summariser.cluster_count_values.loc["A"][CLUSTER_ID_FIELD] == 3
        assert dataset_summariser.cluster_count_values.loc["B"][CLUSTER_ID_FIELD] == 1
        assert dataset_summariser.cluster_counts_weights == {1: 1, 3: 1}

    def test_dataset_summariser_correctly_counts_cluster_links(self, dataset_summariser):
        assert dataset_summariser.weights_for_cluster_sizes == {3: {2: 1, 1: 2}, 1: {2: 1}}
