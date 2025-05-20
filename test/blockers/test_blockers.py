from collections import defaultdict

import pandas as pd
import pytest

from am_combiner.blockers.common import FeatureBasedNameBlocker, FeatureBasedNameBlockerWithCutoff
from am_combiner.combiners.common import CLUSTER_ID_FIELD, BLOCKING_FIELD_FIELD
from am_combiner.features.sanction import Sanction


class TestBlockers:
    @pytest.mark.parametrize(
        ["sanction_types", "sanction_features", "expected_blocks", "expected_blocks_features"],
        [
            (
                ["person", "person", "person"],
                [{"f": [1, 2, 3]}, {"f": [2, 3, 4]}, {"f": [11, 22, 33]}],
                {"person-0", "person-1"},
                {
                    "person-0": [{"f": [1, 2, 3]}, {"f": [2, 3, 4]}],
                    "person-1": [{"f": [11, 22, 33]}],
                },
            ),
            (
                ["person", "person", "vessel"],
                [{"f": [1, 2, 3]}, {"f": [11, 22, 33]}, {"f": [11, 22, 33]}],
                {"person-0", "person-1", "vessel-0"},
                {
                    "person-0": [{"f": [1, 2, 3]}],
                    "person-1": [{"f": [11, 22, 33]}],
                    "vessel-0": [{"f": [11, 22, 33]}],
                },
            ),
        ],
    )
    def test_feature_based_name_blocker(
        self, sanction_types, sanction_features, expected_blocks, expected_blocks_features
    ):
        sanctions = defaultdict(list)
        for sid, (s_type, s_feature) in enumerate(zip(sanction_types, sanction_features)):
            s = Sanction(sanction_id=str(sid), raw_entity={}, sanction_type=s_type)
            s.extracted_entities = s_feature
            sanctions[s_type].append(s)
        blocker = FeatureBasedNameBlocker(use_features=["f"])
        blocked = blocker.block_data(sanctions)
        assert blocked.keys() == expected_blocks
        for k, records in blocked.items():
            assert len(records) == len(expected_blocks_features[k])
            for r, er in zip(records, expected_blocks_features[k]):
                assert r.extracted_entities == er

    @pytest.mark.parametrize(
        ["df_source", "deblock_mapping", "expected_output_df_source"],
        [
            (
                [
                    {BLOCKING_FIELD_FIELD: "person", CLUSTER_ID_FIELD: 0},
                    {BLOCKING_FIELD_FIELD: "person", CLUSTER_ID_FIELD: 0},
                    {BLOCKING_FIELD_FIELD: "person", CLUSTER_ID_FIELD: 0},
                ],
                {"person": "person"},
                [
                    {BLOCKING_FIELD_FIELD: "person", CLUSTER_ID_FIELD: 0},
                    {BLOCKING_FIELD_FIELD: "person", CLUSTER_ID_FIELD: 0},
                    {BLOCKING_FIELD_FIELD: "person", CLUSTER_ID_FIELD: 0},
                ],
            ),
            (
                [
                    {BLOCKING_FIELD_FIELD: "person-0", CLUSTER_ID_FIELD: 0},
                    {BLOCKING_FIELD_FIELD: "person-0", CLUSTER_ID_FIELD: 0},
                    {BLOCKING_FIELD_FIELD: "person-1", CLUSTER_ID_FIELD: 0},
                    {BLOCKING_FIELD_FIELD: "vessel-0", CLUSTER_ID_FIELD: 0},
                ],
                {"person-0": "person", "person-1": "person", "vessel-0": "vessel"},
                [
                    {BLOCKING_FIELD_FIELD: "person", CLUSTER_ID_FIELD: 0},
                    {BLOCKING_FIELD_FIELD: "person", CLUSTER_ID_FIELD: 0},
                    {BLOCKING_FIELD_FIELD: "person", CLUSTER_ID_FIELD: 1},
                    {BLOCKING_FIELD_FIELD: "vessel", CLUSTER_ID_FIELD: 2},
                ],
            ),
        ],
    )
    def test_blocker_data_deblocking(self, df_source, deblock_mapping, expected_output_df_source):
        df = pd.DataFrame(df_source)

        blocker = FeatureBasedNameBlocker(use_features=["f"])
        blocker.deblock_mapping = deblock_mapping
        deblocked = blocker.deblock_labels(df)
        assert deblocked.to_dict(orient="rows") == expected_output_df_source


@pytest.mark.parametrize("cluster_ids", [[1, 1, 1, 1, 2, 2, 2, 3, 4, 5]])
def test_blocker_with_cutoff_block_over_cutoff_clusters(cluster_ids):
    id_to_record = {}
    for i in range(len(cluster_ids)):
        id_to_record[str(i)] = Sanction(str(i), {}, "")
    blocker = FeatureBasedNameBlockerWithCutoff(use_features=[], cluster_cutoff=3, th_ls=[])
    id_to_record = blocker._block_over_cutoff_clusters(cluster_ids, id_to_record, "")
    assert len(id_to_record) == 3
    assert blocker.block_num == 2
