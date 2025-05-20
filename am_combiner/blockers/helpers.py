from am_combiner.blockers.mapping import BLOCKERS_CLASS_MAPPING
from am_combiner.utils.parametrization import get_cache_from_yaml, features_str_to_enum


def get_blockers_cache(blocker_name: str, config_path: str):
    """Small helper function to load all blockers."""
    return get_cache_from_yaml(
        config_path,
        section_name="blockers",
        class_mapping=BLOCKERS_CLASS_MAPPING,
        restrict_classes={blocker_name},
        attrs_callbacks={"use_features": lambda fs: [features_str_to_enum(f) for f in fs]},
    )[blocker_name]
