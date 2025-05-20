from tqdm import tqdm

from am_combiner.utils.data import AbstractInputDataProvider, batch_iterable
from typing import Dict, List
import pandas as pd
from pymongo import MongoClient

from sklearn.model_selection import train_test_split
from am_combiner.combiners.common import (
    UNIQUE_ID_FIELD,
    PROFILE_ID_FIELD,
    SANCTION_ENTITY_FIELD,
    GROUND_TRUTH_FIELD,
    TRAIN_TEST_VALIDATE_SPLIT_FIELD,
    SANCTION_TYPE_FIELD,
    BLOCKING_FIELD_FIELD,
)

MISSING_ENTITY_TYPE = "undefined"


class ManualOverlayUnifyGroundTruth(AbstractInputDataProvider):

    """Implements a data provider which gets ground truth data from Mongo."""

    MONGO_BATCH_SIZE = 50000
    REQUIRED_ATTRIBUTES = [
        "mongo_uri",
        "mongo_database",
        "sm_collection",
        "mo_collection",
        "profile_collection",
        "test_prop",
        "valid_prop",
        "entity_types",
        "sm_types",
        "full_dataset",
    ]

    def __init__(
        self,
        params: Dict,
        entity_names: List[str] = None,
        excluded_entity_names: List[str] = None,
        max_names=None,
        min_content_length=None,
        max_content_length=None,
    ) -> None:
        super().__init__(
            params,
            entity_names,
            excluded_entity_names,
            max_names,
            min_content_length,
            max_content_length,
        )
        self.sm_collection = self.params["sm_collection"]
        self.mo_collection = self.params["mo_collection"]
        self.profile_collection = self.params["profile_collection"]
        self.full_dataset = self.params["full_dataset"]
        self.sm_df = None
        self.mo_df = None
        self.pp_df = None
        self.mongo_db = None
        self.entity_types = self.params["entity_types"]

    def _get_manual_overrides(self) -> List[str]:
        """Filter to sanctions that have been manually unified."""
        query = {"entity.data.hidden_fields.value": "unify", "event_type": "update"}
        return_fields = {"entity.data.display_fields.value": 1, "_id": 0}

        mos = list(self.mongo_db[self.mo_collection].find(query, return_fields))
        mos = [mo["entity"]["data"]["display_fields"][0]["value"] for mo in mos]

        return mos

    def _find_mo_primary_profiles(self, mos) -> List[str]:
        """Update sanction_df with primary profile."""
        query = {
            "entity.data.aml_types.aml_type": {"$in": self.params["sm_types"]},
        }
        if mos is not None:
            query["entity_id"] = {"$in": mos}

        return_fields = {"entity.meta.source_entity_ids": 1, "entity_id": 1, "_id": 0}

        profiles = list(self.mongo_db[self.profile_collection].find(query, return_fields))
        self.pp_df = pd.DataFrame(data=profiles)

        self.pp_df["sm"] = self.pp_df.entity.apply(
            lambda entity: set(entity["meta"]["source_entity_ids"])
        )
        return list(set.union(*self.pp_df.sm.tolist()))

    def _get_mo_structured_mentions(self, primary_profiles) -> None:
        """Get all sanction structured mentions (sm) in mongoDB."""
        query = {
            "entity.data.aml_types.aml_type": {"$in": self.params["sm_types"]},
            "entity_id": {"$in": primary_profiles},
        }
        return_fields = {SANCTION_ENTITY_FIELD: 1, "entity_id": 1, "_id": 0}

        sms = list(self.mongo_db[self.sm_collection].find(query, return_fields))
        if self.sm_df is None:
            self.sm_df = pd.DataFrame(data=sms)
        else:
            self.sm_df = self.sm_df.append(sms)

    def _train_test_validate_split(self, seed: int = 0) -> None:
        """Split data into train/test/validate bunches."""
        assert self.params["test_prop"] + self.params["valid_prop"] <= 1
        unique_profiles = list(self.sm_df.profile_id.unique())

        unique_profiles, test_profiles = train_test_split(
            unique_profiles, test_size=self.params["test_prop"], random_state=seed
        )
        _, valid_profiles = train_test_split(
            unique_profiles,
            test_size=self.params["valid_prop"] / (1 - self.params["test_prop"]),
            random_state=seed,
        )
        valid_profiles, test_profiles = set(valid_profiles), set(test_profiles)
        self.sm_df[TRAIN_TEST_VALIDATE_SPLIT_FIELD] = self.sm_df.profile_id.apply(
            lambda p: "test" if p in test_profiles else "valid" if p in valid_profiles else "train"
        )

    def _enrich_sm_data(self):
        # add sm -> profile mapping:
        sm_to_profile = dict()
        for profile, sms in zip(self.pp_df.entity_id.tolist(), self.pp_df.sm.tolist()):
            for sm in sms:
                sm_to_profile[sm] = profile

        self.sm_df[PROFILE_ID_FIELD] = self.sm_df[UNIQUE_ID_FIELD].apply(
            lambda sm_id: sm_to_profile[sm_id] if sm_id in sm_to_profile else None
        )

        self.sm_df.sort_values(PROFILE_ID_FIELD, inplace=True)

        # add ground_truth based on mo profiles:
        self.sm_df[GROUND_TRUTH_FIELD] = self.sm_df[PROFILE_ID_FIELD].astype("category").cat.codes
        # Add entity type:
        self.sm_df[SANCTION_TYPE_FIELD] = self.sm_df[SANCTION_ENTITY_FIELD].apply(
            lambda entity: self._get_entity_type(entity)
        )

        if self.entity_types:
            mask = self.sm_df[SANCTION_TYPE_FIELD].isin(self.entity_types)
            self.sm_df = self.sm_df[mask]

        self.sm_df[BLOCKING_FIELD_FIELD] = self.sm_df[SANCTION_TYPE_FIELD]

    @staticmethod
    def _get_entity_type(raw_entity) -> str:
        if raw_entity["data"]["entity_types"] is None:
            return MISSING_ENTITY_TYPE
        else:
            return raw_entity["data"]["entity_types"][0]["entity_type"]

    def _get_dataframe(self):
        return

    def get_dataframe(self) -> pd.DataFrame:
        """Query collections to retrieve ground truth from manually unified profiles."""
        self.mongo_db = MongoClient(self.params["mongo_uri"])[self.params["mongo_database"]]

        if self.full_dataset:
            manual_overrides = None
        else:
            manual_overrides = self._get_manual_overrides()

        primary_profiles = self._find_mo_primary_profiles(manual_overrides)

        primary_profiles_batches = batch_iterable(
            primary_profiles, ManualOverlayUnifyGroundTruth.MONGO_BATCH_SIZE
        )
        for batch in tqdm(primary_profiles_batches):
            self._get_mo_structured_mentions(batch)
        self.sm_df.rename(columns={"entity_id": UNIQUE_ID_FIELD}, inplace=True)
        self.sm_df.reset_index(inplace=True)
        self._enrich_sm_data()

        self._train_test_validate_split()
        return self.sm_df
