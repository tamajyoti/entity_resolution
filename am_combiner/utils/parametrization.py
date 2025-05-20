from typing import Union, Dict, Tuple

import yaml

from am_combiner.features.article import Features
from am_combiner.features.sanction import SanctionFeatures


def features_str_to_enum(feature: str) -> Union[Features, str]:
    """
    Attempt to convert a string to one of Features enum values.

    Only strings started with Feature.SOME_STRING_GOES_HERE will be attempted to be converted.
    Otherwise, the input is returned as is.

    Parameters
    ----------
    feature:
        A string representation that needs to be converted.

    Returns
    -------
        Either Features enum values or the original string.

    """
    if feature.startswith("Features.") and len(feature.split(".")) == 2:
        return Features[feature.split(".")[-1]]
    elif feature.startswith("SanctionFeatures.") and len(feature.split(".")) == 2:
        return SanctionFeatures[feature.split(".")[-1]]
    return feature


def get_cache_from_yaml(
    yaml_path: str,
    section_name: str,
    class_mapping: Dict,
    restrict_classes: Union[set, None] = None,
    attrs_callbacks=None,
) -> Dict:
    """
    Convert a yaml section into combiners/visitors objects, constructed from a mapping.

    Parameters
    ----------
    restrict_classes:
        A set of classes which should be constructed.
        Other classes will not be created by default.
    yaml_path:
        Path to read a yaml from.
    section_name:
        Top-level section name of the yaml files.
    class_mapping:
        A dictionary specifying how to convert class references from yaml to actual class objects.
    attrs_callbacks:
        Special callbacks that need to be run on certain attrs values.

    Returns
    -------
        Objects constructed from the yaml.

    """
    if attrs_callbacks is None:
        attrs_callbacks = {}

    output_cache = {}
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
        for object_ref in config[section_name]:
            object_item: Tuple[str, Dict] = list(object_ref.items())[0]
            object_name, object_config = object_item
            if restrict_classes and object_name not in restrict_classes:
                continue

            class_reference: str = object_config["class"]
            object_class = class_mapping.get(class_reference)
            if object_class is None:
                raise Exception(
                    f"Class {class_reference} does not exist in the class mapping "
                    f"for {section_name}"
                )
            attrs = object_config["attrs"] if object_config.get("attrs") else {}
            for attr in attrs:
                if attr not in attrs_callbacks:
                    continue
                attrs[attr] = attrs_callbacks[attr](attrs[attr])
            try:
                output_cache[object_name] = object_class(**attrs)
            except Exception as ex:
                # If literally anything goes wrong during init
                # (e.g. path were incorrect a in yaml file, we report it
                print(
                    f"Failed to initialise {object_name} with class {class_reference}, "
                    f"tried attrs: {attrs}, due to {ex}"
                )
    return output_cache
