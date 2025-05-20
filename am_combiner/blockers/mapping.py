from am_combiner.blockers.common import (
    IdentityBlocker,
    FeatureBasedNameBlocker,
    FeatureBasedNameBlockerWithCutoff,
)
from am_combiner.blockers.country import CountryBlocker

BLOCKERS_CLASS_MAPPING = {
    "IdentityBlocker": IdentityBlocker,
    "FeatureBasedNameBlocker": FeatureBasedNameBlocker,
    "FeatureBasedNameBlockerWithCutoff": FeatureBasedNameBlockerWithCutoff,
    "CountryBlocker": CountryBlocker,
}
