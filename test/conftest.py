import io

import pandas as pd
import pytest

from am_combiner.combiners.common import (
    ENTITY_NAME_FIELD,
    CLUSTER_ID_FIELD,
    URL_FIELD,
    TEXT_COLUMN_FIELD,
    BLOCKING_FIELD_FIELD,
    UNIQUE_ID_FIELD,
    CLUSTER_NUMBER_FIELD,
    GROUND_TRUTH_FIELD,
)
from am_combiner.features.article import Article
from am_combiner.features.geography import (
    CapitalAdditionVisitor,
    CountyAdditionGraphBasedGeoResolverVisitor,
    CountriesCodesVisitor,
    CountriesAliasesGraphBasedGeoResolverVisitor,
    CountriesListGraphBasedGeoResolverVisitor,
    GraphBasedGeoResolver,
)
from am_combiner.utils.distributions import DataframeDistributionSummariser


@pytest.fixture()
def article():
    def article_factory(entity_name, text_block):
        return Article(entity_name, article_text=text_block)

    return article_factory


@pytest.fixture
def entity_names():
    return ["A", "B", "C"]


@pytest.fixture
def excluded_entity_names():
    return ["D", "E"]


@pytest.fixture
def test_dataframe_as_csv_content():
    return io.StringIO(
        ",blocking_field,content,unique_id\n"
        "0,A,AA,http://A\n"
        "1,B,BB,http://B\n"
        "2,C,CC,http://C\n"
        "3,D,DD,http://D\n"
        "4,E,EE,http://E\n"
    )


@pytest.fixture
def test_dataframe():
    return pd.DataFrame(
        [
            {
                ENTITY_NAME_FIELD: "A",
                CLUSTER_ID_FIELD: 0,
                URL_FIELD: "http1",
                TEXT_COLUMN_FIELD: "text1",
            },
            {
                ENTITY_NAME_FIELD: "A",
                CLUSTER_ID_FIELD: 0,
                URL_FIELD: "http2",
                TEXT_COLUMN_FIELD: "text2",
            },
            {
                ENTITY_NAME_FIELD: "A",
                CLUSTER_ID_FIELD: 1,
                URL_FIELD: "http3",
                TEXT_COLUMN_FIELD: "text3",
            },
            {
                ENTITY_NAME_FIELD: "A",
                CLUSTER_ID_FIELD: 2,
                URL_FIELD: "http4",
                TEXT_COLUMN_FIELD: "text4",
            },
            {
                ENTITY_NAME_FIELD: "B",
                CLUSTER_ID_FIELD: 0,
                URL_FIELD: "http5",
                TEXT_COLUMN_FIELD: "text5",
            },
            {
                ENTITY_NAME_FIELD: "B",
                CLUSTER_ID_FIELD: 0,
                URL_FIELD: "http6",
                TEXT_COLUMN_FIELD: "text6",
            },
        ]
    )


@pytest.fixture
def dataset_summariser(test_dataframe):
    return DataframeDistributionSummariser(test_dataframe)


@pytest.fixture
def full_geo_resolver():
    geo_resolver = GraphBasedGeoResolver()
    country_list_visitor = CountriesListGraphBasedGeoResolverVisitor(
        country_list_fn="am_combiner/data/geo/all_countries.csv"
    )
    geo_resolver.accept_visitor(country_list_visitor)

    country_aliases_visitor = CountriesAliasesGraphBasedGeoResolverVisitor(
        country_aliases_fn="am_combiner/data/geo/countries_alternative_names.tsv"
    )
    geo_resolver.accept_visitor(country_aliases_visitor)

    countries_codes_visitor = CountriesCodesVisitor(
        country_list_fn="am_combiner/data/geo/all_countries.csv"
    )
    geo_resolver.accept_visitor(countries_codes_visitor)

    county_addition_visitor = CountyAdditionGraphBasedGeoResolverVisitor(
        resource_path="am_combiner/data/geo/"
    )
    geo_resolver.accept_visitor(county_addition_visitor)

    capital_addition_visitor = CapitalAdditionVisitor(resource_path="am_combiner/data/geo/")
    geo_resolver.accept_visitor(capital_addition_visitor)
    return geo_resolver


@pytest.fixture
def entity_name():
    return "Some Name"


@pytest.fixture
def other_entity_name():
    return "Some OTHER Name"


@pytest.fixture
def validation(entity_name, other_entity_name):
    return pd.DataFrame(
        [
            {BLOCKING_FIELD_FIELD: entity_name, UNIQUE_ID_FIELD: "url.1", GROUND_TRUTH_FIELD: 1},
            {BLOCKING_FIELD_FIELD: entity_name, UNIQUE_ID_FIELD: "url.2", GROUND_TRUTH_FIELD: 2},
            {BLOCKING_FIELD_FIELD: entity_name, UNIQUE_ID_FIELD: "url.3", GROUND_TRUTH_FIELD: 2},
            {BLOCKING_FIELD_FIELD: entity_name, UNIQUE_ID_FIELD: "url.4", GROUND_TRUTH_FIELD: 3},
            {BLOCKING_FIELD_FIELD: entity_name, UNIQUE_ID_FIELD: "url.5", GROUND_TRUTH_FIELD: 4},
            {BLOCKING_FIELD_FIELD: entity_name, UNIQUE_ID_FIELD: "url.6", GROUND_TRUTH_FIELD: 4},
            {BLOCKING_FIELD_FIELD: entity_name, UNIQUE_ID_FIELD: "url.7", GROUND_TRUTH_FIELD: 4},
            {
                BLOCKING_FIELD_FIELD: other_entity_name,
                UNIQUE_ID_FIELD: "url.1",
                GROUND_TRUTH_FIELD: 1,
            },
            {
                BLOCKING_FIELD_FIELD: other_entity_name,
                UNIQUE_ID_FIELD: "url.2",
                GROUND_TRUTH_FIELD: 2,
            },
        ]
    )


@pytest.fixture
def clustering_results(entity_name, other_entity_name):
    return pd.DataFrame(
        [
            {
                BLOCKING_FIELD_FIELD: entity_name,
                UNIQUE_ID_FIELD: "url.3",
                CLUSTER_NUMBER_FIELD: 3,
            },
            {
                BLOCKING_FIELD_FIELD: entity_name,
                UNIQUE_ID_FIELD: "url.1",
                CLUSTER_NUMBER_FIELD: 1,
            },
            {
                BLOCKING_FIELD_FIELD: entity_name,
                UNIQUE_ID_FIELD: "url.2",
                CLUSTER_NUMBER_FIELD: 2,
            },
            {
                BLOCKING_FIELD_FIELD: entity_name,
                UNIQUE_ID_FIELD: "url.4",
                CLUSTER_NUMBER_FIELD: 1,
            },
            {
                BLOCKING_FIELD_FIELD: entity_name,
                UNIQUE_ID_FIELD: "url.5",
                CLUSTER_NUMBER_FIELD: 4,
            },
            {
                BLOCKING_FIELD_FIELD: entity_name,
                UNIQUE_ID_FIELD: "url.6",
                CLUSTER_NUMBER_FIELD: 4,
            },
            {
                BLOCKING_FIELD_FIELD: entity_name,
                UNIQUE_ID_FIELD: "url.7",
                CLUSTER_NUMBER_FIELD: 5,
            },
            {
                BLOCKING_FIELD_FIELD: other_entity_name,
                UNIQUE_ID_FIELD: "url.1",
                CLUSTER_NUMBER_FIELD: 1,
            },
            {
                BLOCKING_FIELD_FIELD: other_entity_name,
                UNIQUE_ID_FIELD: "url.2",
                CLUSTER_NUMBER_FIELD: 2,
            },
        ]
    )


@pytest.fixture
def expected_clustering(entity_name):
    return pd.DataFrame(
        [
            {BLOCKING_FIELD_FIELD: entity_name, UNIQUE_ID_FIELD: "url.1", GROUND_TRUTH_FIELD: 1},
            {BLOCKING_FIELD_FIELD: entity_name, UNIQUE_ID_FIELD: "url.2", GROUND_TRUTH_FIELD: 2},
            {BLOCKING_FIELD_FIELD: entity_name, UNIQUE_ID_FIELD: "url.3", GROUND_TRUTH_FIELD: 2},
            {BLOCKING_FIELD_FIELD: entity_name, UNIQUE_ID_FIELD: "url.4", GROUND_TRUTH_FIELD: 3},
            {BLOCKING_FIELD_FIELD: entity_name, UNIQUE_ID_FIELD: "url.5", GROUND_TRUTH_FIELD: 4},
            {BLOCKING_FIELD_FIELD: entity_name, UNIQUE_ID_FIELD: "url.6", GROUND_TRUTH_FIELD: 4},
            {BLOCKING_FIELD_FIELD: entity_name, UNIQUE_ID_FIELD: "url.7", GROUND_TRUTH_FIELD: 4},
        ]
    )


@pytest.fixture
def actual_clustering(entity_name):
    return pd.DataFrame(
        [
            {
                BLOCKING_FIELD_FIELD: entity_name,
                UNIQUE_ID_FIELD: "url.1",
                CLUSTER_NUMBER_FIELD: 1,
            },
            {
                BLOCKING_FIELD_FIELD: entity_name,
                UNIQUE_ID_FIELD: "url.2",
                CLUSTER_NUMBER_FIELD: 2,
            },
            {
                BLOCKING_FIELD_FIELD: entity_name,
                UNIQUE_ID_FIELD: "url.3",
                CLUSTER_NUMBER_FIELD: 3,
            },
            {
                BLOCKING_FIELD_FIELD: entity_name,
                UNIQUE_ID_FIELD: "url.4",
                CLUSTER_NUMBER_FIELD: 1,
            },
            {
                BLOCKING_FIELD_FIELD: entity_name,
                UNIQUE_ID_FIELD: "url.5",
                CLUSTER_NUMBER_FIELD: 4,
            },
            {
                BLOCKING_FIELD_FIELD: entity_name,
                UNIQUE_ID_FIELD: "url.6",
                CLUSTER_NUMBER_FIELD: 4,
            },
            {
                BLOCKING_FIELD_FIELD: entity_name,
                UNIQUE_ID_FIELD: "url.7",
                CLUSTER_NUMBER_FIELD: 5,
            },
        ]
    )
