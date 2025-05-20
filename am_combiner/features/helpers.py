from typing import Optional, Set

from am_combiner.features.mapping import VISITORS_CLASS_MAPPING
from am_combiner.utils.parametrization import features_str_to_enum, get_cache_from_yaml


def get_visitors_cache(visitors: Optional[Set], config_path: str):
    """Small helper function to load all visitors."""
    return get_cache_from_yaml(
        config_path,
        section_name="visitors",
        class_mapping=VISITORS_CLASS_MAPPING,
        restrict_classes=visitors,
        attrs_callbacks={
            "feature_name": features_str_to_enum,
            "source_feature": features_str_to_enum,
            "target_feature": features_str_to_enum,
            "field_name": features_str_to_enum,
            "feature_topic": features_str_to_enum,
            "feature_distribution": features_str_to_enum,
            "feature_tfidf": features_str_to_enum,
            "use_features": lambda fs: [features_str_to_enum(f) for f in fs],
            "feature_name_yob": features_str_to_enum,
            "feature_name_dob": features_str_to_enum,
            "bypass_translation": bool,
            "feature_name_yob_known": features_str_to_enum,
        },
    )
