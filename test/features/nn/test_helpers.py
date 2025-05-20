import pytest

from am_combiner.combiners.common import CLUSTER_ID_GLOBAL_FIELD, UNIQUE_ID_FIELD
from am_combiner.features.nn.helpers import build_cluster_id_cache
import pandas as pd


class TestHelpers:
    @pytest.mark.parametrize(
        ["dataframe", "expected"],
        [
            (
                pd.DataFrame(
                    {
                        CLUSTER_ID_GLOBAL_FIELD: [1, 2, 3],
                        UNIQUE_ID_FIELD: ["http1", "http2", "http3"],
                    }
                ),
                {"http1": 1, "http2": 2, "http3": 3},
            ),
            (
                pd.DataFrame(
                    {
                        CLUSTER_ID_GLOBAL_FIELD: [1, 2, 3, 1],
                        UNIQUE_ID_FIELD: ["http1", "http2", "http3", "http4"],
                    }
                ),
                {"http1": 1, "http2": 2, "http3": 3, "http4": 1},
            ),
        ],
    )
    def test_index_building(self, dataframe, expected):
        output = build_cluster_id_cache(dataframe)
        assert output == expected

    @pytest.mark.parametrize(
        [
            "dataframe",
        ],
        [
            (pd.DataFrame({UNIQUE_ID_FIELD: ["http1"], "field1": [1]}),),
            (pd.DataFrame({"field1": ["http1"], CLUSTER_ID_GLOBAL_FIELD: [1]}),),
            (pd.DataFrame({"field1": ["http1"], "field2": [1]}),),
        ],
    )
    def test_does_not_proceed_without_required_fields(self, dataframe):
        with pytest.raises(ValueError):
            build_cluster_id_cache(dataframe)

    def test_fails_with_contradicting_ids(self):
        df = pd.DataFrame({UNIQUE_ID_FIELD: ["http1", "http1"], CLUSTER_ID_GLOBAL_FIELD: [1, 2]})
        with pytest.raises(ValueError):
            build_cluster_id_cache(df)
