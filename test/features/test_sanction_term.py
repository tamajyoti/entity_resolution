import pytest

from am_combiner.features.sanction import Sanction, SanctionFeatures
from am_combiner.features.sanction_term import SanctionTermVisitor, SanctionTermSpacyVisitor


class TestSanctionTermExtraction:
    @pytest.mark.parametrize(
        [
            "raw_sanction",
            "expected_url",
            "expected_function",
            "expected_other_information",
            "expected_norp",
        ],
        [
            (
                {
                    "data": {
                        "display_fields": [
                            {
                                "asset_id": None,
                                "justification": {
                                    "asset_ids": [],
                                    "field_ids": [],
                                    "score": 1.0,
                                    "source_ids": ["S:29A07C"],
                                },
                                "language": None,
                                "original_language": None,
                                "original_value": None,
                                "rank": 5,
                                "title": "Function",
                                "value": "commandant de la 155ème brigade de missiles",
                            },
                            {
                                "asset_id": None,
                                "justification": {
                                    "asset_ids": [],
                                    "field_ids": [],
                                    "score": 1.0,
                                    "source_ids": ["S:29A07C"],
                                },
                                "language": None,
                                "original_language": None,
                                "original_value": None,
                                "rank": 5,
                                "title": "Related Url",
                                "value": "https://www.tresor.economie.gouv.fr/",
                            },
                            {
                                "asset_id": None,
                                "justification": {
                                    "asset_ids": [],
                                    "field_ids": [],
                                    "score": 1.0,
                                    "source_ids": ["S:29A07C"],
                                },
                                "language": None,
                                "original_language": None,
                                "original_value": None,
                                "rank": 5,
                                "title": "Other Information",
                                "value": "Additional information:member of the Syrian armed forces",
                            },
                        ]
                    }
                },
                {"https://www.tresor.economie.gouv.fr/"},
                {"commander of the 155th missile brigade"},
                {"Additional information:member of the Syrian armed forces"},
                {"syrian"},
            ),
        ],
    )
    def test_sanction_term(
        self,
        raw_sanction,
        expected_url,
        expected_function,
        expected_other_information,
        expected_norp,
    ):
        s = Sanction("", raw_sanction, "person")
        v = SanctionTermVisitor(
            use_features=[
                SanctionFeatures.RELATED_URL,
                SanctionFeatures.FUNCTION,
                SanctionFeatures.OTHER_INFORMATION,
            ],
            bypass_translation=False,
        )
        s.accept_visitor(v)

        assert s.extracted_entities[SanctionFeatures.RELATED_URL] == expected_url
        assert s.extracted_entities[SanctionFeatures.FUNCTION] == expected_function
        assert (
            s.extracted_entities[SanctionFeatures.OTHER_INFORMATION] == expected_other_information
        )

        v = SanctionTermSpacyVisitor(use_features=[SanctionFeatures.OTHER_INFORMATION])
        s.accept_visitor(v)
        assert s.extracted_entities[SanctionFeatures.NORP] == expected_norp

        return s
