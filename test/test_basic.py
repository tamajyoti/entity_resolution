import pytest

from am_combiner.features.article import Features
from am_combiner.utils.parametrization import features_str_to_enum


class TestFeatureToStrEnumeration:
    @pytest.mark.parametrize(
        "string_to_convert,expected_type,expected_value",
        [
            ("Features.PERSON", Features, Features.PERSON),
            ("NOTFeatures.PERSON", str, "NOTFeatures.PERSON"),
            ("PERSON", str, "PERSON"),
        ],
    )
    def test_converting_feature_string_to_enum(
        self, string_to_convert, expected_type, expected_value
    ):
        res = features_str_to_enum(string_to_convert)
        assert isinstance(res, expected_type)
        assert res == expected_value
