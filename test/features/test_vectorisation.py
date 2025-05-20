import pytest

from am_combiner.features.sanction import Sanction, SanctionFeatures
from am_combiner.features.vectorisation import JsonSummarizer


class TestVectorisation:
    @pytest.mark.parametrize(
        ["raw_entity", "expected"],
        [
            (
                {
                    "data": {
                        "display_fields": [
                            {"title": "Other Information", "value": "This is some other info"},
                            {"title": "Function", "value": "Totally hopeless"},
                        ]
                    }
                },
                "This is some other info.Totally hopeless",
            ),
        ],
    )
    def test_full_text_formation(self, raw_entity, expected):
        s = Sanction(sanction_id="A", raw_entity=raw_entity, sanction_type="person")
        js = JsonSummarizer()
        js.visit_sanction(s)
        assert s.extracted_entities[SanctionFeatures.FULL_TEXT] == expected
