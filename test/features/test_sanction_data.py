import pandas as pd
from am_combiner.utils.sanction_data import ManualOverlayUnifyGroundTruth
from am_combiner.combiners.common import TRAIN_TEST_VALIDATE_SPLIT_FIELD


def test_train_test_validate_split():
    provider = ManualOverlayUnifyGroundTruth(
        {
            "mongo_uri": "",
            "mongo_database": "",
            "sm_collection": "",
            "mo_collection": "",
            "profile_collection": "",
            "entity_types": "",
            "sm_types": "",
            "test_prop": 0.2,
            "valid_prop": 0.5,
            "full_dataset": False,
        }
    )
    provider.sm_df = pd.DataFrame(data={"profile_id": list(range(1001))})
    provider._train_test_validate_split()

    test_size = (provider.sm_df[TRAIN_TEST_VALIDATE_SPLIT_FIELD] == "test").mean()
    valid_size = (provider.sm_df[TRAIN_TEST_VALIDATE_SPLIT_FIELD] == "valid").mean()

    assert 0.19 < test_size < 0.21
    assert 0.49 < valid_size < 0.51
