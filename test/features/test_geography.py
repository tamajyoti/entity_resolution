import pytest


class TestGeographicalResolution:
    @pytest.mark.parametrize(
        ["country_name", "expected_output"],
        [
            # This group is for basic testing
            ("United Kingdom", ["united kingdom"]),
            ("Afghanistan", ["afghanistan"]),
            ("Australia", ["australia"]),
            # This group is for country alternative names resolution
            ("The United Kingdom", ["united kingdom"]),
            ("UK", ["united kingdom"]),
            ("U.K.", ["united kingdom"]),
            # This group is for testing country codes resolutions
            ("UK", ["united kingdom"]),
            ("RU", ["russian federation"]),
            ("US", ["united states"]),
            # This group is for resolving counties/state level entities
            ("texas", ["united states"]),
            # This group is for capital resolution
            ("london", ["united kingdom"]),
            ("moscow", ["russian federation"]),
        ],
    )
    def test_simple_country_name_resolution(self, full_geo_resolver, country_name, expected_output):
        resolved = full_geo_resolver.resolve_geo_name(country_name, {"final": True})
        assert resolved == expected_output
