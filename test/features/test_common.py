import pytest

from am_combiner.features.common import (
    TextCleaner,
    remove_self_reference,
    convert_name_to_keyword_tokens,
    SanctionPrimariesExtractor,
    SanctionAliasExtractor,
    SanctionPassportVisitor,
    NationalityVisitor,
    SanctionBirthExtractor,
    get_phonetic_keyword,
    FathersNamesFromAlias,
)
from am_combiner.features.sanction import Sanction, SanctionFeatures


class TestRemoveSelfReference:
    @pytest.mark.parametrize(
        ["names_list", "entity_name", "expected_output"],
        [
            ({"John Smith", "Marble Arch"}, "John", {"Marble Arch"}),
            ({"John Smith", "Marble Arch"}, "John Marble", set()),
            ({"John Smith", "Marble Arch"}, "Tim", {"John Smith", "Marble Arch"}),
            ({"John Smith", "Marble Arch"}, "Ar", {"John Smith", "Marble Arch"}),
            ({"John Smith", "Marble Arch"}, "MIT", {"John Smith", "Marble Arch"}),
            ({"John Smith", "Marble Arch"}, "Fred S. Johnson", {"John Smith", "Marble Arch"}),
            ({"John Smith", "Marble S. Arch"}, "Fred S. Johnson", {"John Smith", "Marble S. Arch"}),
            ({"John Smith", "Marble Leed Arch"}, "Lee.", {"John Smith", "Marble Leed Arch"}),
            ({"Joe Doe", "joe", "joe meme"}, "joe", set()),
            (
                {
                    "Manhattan",
                    "Apalachi",
                },
                "a",
                {"Manhattan", "Apalachi"},
            ),
        ],
    )
    def test_remove_self_reference(self, names_list, entity_name, expected_output):
        assert remove_self_reference(names_list, entity_name) == expected_output

    def test_remove_self_reference_keeps_type(self):
        assert type(remove_self_reference(["A", "B"], "A")) == list
        assert type(remove_self_reference({"A", "B"}, "A")) == set

    def test_remove_self_reference_fails_with_other_types(self):
        with pytest.raises(ValueError):
            remove_self_reference("AA", "A")


class TestTextCleaner:
    @pytest.mark.parametrize(
        ["input_text", "expected_text"],
        [
            ("<html>Hello world</html>", "Hello world"),
            ("<html></html>", ""),
            ("<html><br><br>Hello world</html>", "Hello world"),
            ("Hello world</html>", "Hello world"),
        ],
    )
    def test_html_tag_removed(self, input_text, expected_text):
        cleaned_text = TextCleaner().clean_html_tag(input_text)
        assert cleaned_text == expected_text

    @pytest.mark.parametrize(
        ["input_text", "expected_text"],
        [
            ("Hello world", "Hello world"),
            ("<<<<<<Hello world", "<<<<<<Hello world"),
        ],
    )
    def test_text_unmodified_if_no_tags(self, input_text, expected_text):
        cleaned_text = TextCleaner().clean_html_tag(input_text)
        assert cleaned_text == expected_text


class TestSanctionNames:
    @pytest.mark.parametrize(
        ["input_text", "expected_output"],
        [
            ("Les défenseurs de la main rouge", ["defenseurs", "les", "main", "rouge"]),
            ("Ur-Rehman Abd", ["abd", "rehman"]),
            ("Nizar (al) Assad", ["assad", "nizar"]),
            ("Sa-id", ["sa-id"]),
            ("Vladimir Putin;President", ["president", "putin", "vladimir"]),
            ("`Boris Johnson", ["boris", "johnson"]),
            ("what's going", ["going", "what"]),
            ("عبدالباقی بصیراول شاه", ["bdlbqy", "bsyrwl", "shh"]),
        ],
    )
    def test_keyword_tokenization(self, input_text, expected_output):
        output_tokens = convert_name_to_keyword_tokens(input_text)
        assert output_tokens == expected_output

    @pytest.mark.parametrize(
        ["input_a", "input_b", "is_match"],
        [
            ("Alexey", "Aleksei", True),
            ("Mohammed", "Mohamed", True),
            ("Alexey", "Andrei", False),
            ("Mohammed", "Mahud", False),
        ],
    )
    def test_phonetic(self, input_a, input_b, is_match):
        phonetic_a = get_phonetic_keyword([input_a])
        phonetic_b = get_phonetic_keyword([input_b])
        assert (phonetic_a == phonetic_b) == is_match

    @pytest.mark.parametrize(
        ["raw_entity", "expected_primary", "expected_kw", "expected_bigrams"],
        [
            (
                {
                    "data": {
                        "names": [
                            {"name": "Er True Corrector Mosby", "name_type": "primary"},
                            {"name": "Le Falsifier", "name_type": "alias"},
                        ]
                    }
                },
                {"Er True Corrector Mosby"},
                {"corrector+mosby+true", "falsifier"},
                {"corrector+true", "corrector+mosby", "mosby+true"},
            ),
        ],
    )
    def test_visit_primary_alias(self, raw_entity, expected_primary, expected_kw, expected_bigrams):
        sanction = Sanction("", raw_entity, "")
        SanctionPrimariesExtractor().visit_sanction(sanction)
        SanctionAliasExtractor().visit_sanction(sanction)

        assert sanction.extracted_entities[SanctionFeatures.PRIMARY] == expected_primary
        assert sanction.extracted_entities[SanctionFeatures.ALIAS_KEYWORD] == expected_kw
        assert (
            sanction.extracted_entities[SanctionFeatures.ALIAS_KEYWORD_BIGRAMS] == expected_bigrams
        )

    @pytest.mark.parametrize(
        ["raw_entity", "expected_bigrams"],
        [
            (
                {"data": {"names": [{"name": "Brigade general Gegole", "name_type": "alias"}]}},
                {"brigade+general", "brigade+gegole", "gegole+general"},
            ),
            (
                {"data": {"names": [{"name": "Brigade", "name_type": "alias"}]}},
                set(),
            ),
        ],
    )
    def test_bigrams_extraction(self, raw_entity, expected_bigrams):
        sanction = Sanction("", raw_entity, "")
        SanctionAliasExtractor().visit_sanction(sanction)
        assert (
            sanction.extracted_entities[SanctionFeatures.ALIAS_KEYWORD_BIGRAMS] == expected_bigrams
        )


class TestSanctionPassportVisitor:
    @pytest.mark.parametrize(
        ["raw_sanction"],
        [
            ({"no_data_jere": ":("},),
            ({"data": {"no_passports_here": ":("}},),
            ({"data": {"passports": None}},),
            ({"data": {"passports": []}},),
        ],
    )
    def test_does_not_fail_on_missing_fields(self, raw_sanction):
        s = Sanction("", raw_sanction, "person")
        v = SanctionPassportVisitor()
        s.accept_visitor(v)

    @pytest.mark.parametrize(
        ["raw_sanction"],
        [
            ({"data": {"passports": [{"passport": "Russia, 25.08.2005"}]}},),
        ],
    )
    def test_extracts_some_basic_info(self, raw_sanction):
        s = Sanction("", raw_sanction, "person")
        v = SanctionPassportVisitor()
        s.accept_visitor(v)
        assert "russia" in s.extracted_entities[SanctionFeatures.PASSPORT_GPE]
        assert s.extracted_entities[SanctionFeatures.PASSPORT_RAW] == ["Russia, 25.08.2005"]

    @pytest.mark.parametrize(
        ["input_passport", "expected"],
        [
            ("W12345 issued by the UK", "w12345"),
            ("issued by the UK in 2015", ""),
            ("W12345 was legit issued by agency RTU56", "rtu56+w12345"),
        ],
    )
    def test_pid_extraction(self, input_passport, expected):
        assert SanctionPassportVisitor._extract_pid(input_passport) == expected


class TestBirthExtractor:
    @pytest.mark.parametrize(
        ["raw_sanction", "expected_yob", "expected_dob"],
        [
            (
                {
                    "data": {
                        "births": [
                            {
                                "justification": {
                                    "asset_ids": [],
                                    "field_ids": [],
                                    "score": 1.0,
                                    "source_ids": ["S:TY9XBH"],
                                },
                                "max_date": "1959-12-31",
                                "min_date": "1959-01-01",
                            },
                            {
                                "justification": {
                                    "asset_ids": [],
                                    "field_ids": [],
                                    "score": 1.0,
                                    "source_ids": ["S:TY9XBH"],
                                },
                                "max_date": "1960-03-31",
                                "min_date": "1960-03-31",
                            },
                            {
                                "justification": {
                                    "asset_ids": [],
                                    "field_ids": [],
                                    "score": 1.0,
                                    "source_ids": ["S:TY9XBH"],
                                },
                                "max_date": "1958-04-29",
                                "min_date": "1958-04-29",
                            },
                        ]
                    }
                },
                {1958, 1959, 1960},
                {"1958-04-29", "1960-03-31"},
            ),
        ],
    )
    def test_yob_dob(self, raw_sanction, expected_yob, expected_dob):
        s = Sanction("", raw_sanction, "person")
        v = SanctionBirthExtractor(
            feature_name_yob=SanctionFeatures.YOB, feature_name_dob=SanctionFeatures.DOB
        )
        s.accept_visitor(v)
        assert s.extracted_entities[SanctionFeatures.YOB] == expected_yob
        assert s.extracted_entities[SanctionFeatures.DOB] == expected_dob

    @pytest.mark.parametrize(
        ["raw_sanction", "expected"],
        [
            (
                {
                    "data": {
                        "births": [
                            {"min_date": "1967-01-01", "max_date": "1967-31-12"},
                            {"min_date": "1968-01-01", "max_date": "1968-12-31"},
                            {"min_date": "1969-01-01", "max_date": "1971-01-01"},
                        ]
                    }
                },
                {"1967", "1968", "1969", "1971"},
            ),
            (
                {
                    "data": {
                        "births": [
                            {"min_date": "1979-01-01", "max_date": "1989-01-01"},
                            {"min_date": "1969-01-01", "max_date": "1971-01-01"},
                        ]
                    }
                },
                {"1969", "1971", "1979", "1989"},
            ),
        ],
    )
    def test_known_yob_extraction(self, raw_sanction, expected):
        s = Sanction("", raw_sanction, "person")
        v = SanctionBirthExtractor()
        s.accept_visitor(v)
        assert s.extracted_entities[SanctionFeatures.YOB_KNOWN] == expected


class TestNationalitiesVisitor:
    @pytest.mark.parametrize(
        ["raw_sanction", "expected"],
        [
            ({"data": {"nationalities": [{"country_code": "AF"}]}}, {"AF"}),
            (
                {"data": {"nationalities": [{"country_code": "RU"}, {"country_code": "UA"}]}},
                {"RU", "UA"},
            ),
        ],
    )
    def test_extracts_country_codes(self, raw_sanction, expected):
        s = Sanction("", raw_sanction, "person")
        v = NationalityVisitor()
        s.accept_visitor(v)
        assert s.extracted_entities[SanctionFeatures.NATIONALITIES] == expected


class TestFathersNamesVisitor:
    @pytest.mark.parametrize(
        ["raw_sanction1", "raw_sanction2", "same_fathers_expected"],
        [
            (
                {"data": {"names": [{"name": "Nazar Hussain s/o Khan Muhammad"}]}},
                {"data": {"names": [{"name": "Nazar Hussain s/o Mohamed Khan"}]}},
                True,
            ),
            (
                {"data": {"names": [{"name": "Riaz Ahmad s/o Bashir Ahmad"}]}},
                {"data": {"names": [{"name": "Ahmad Raza s/o Baiq Raza"}]}},
                False,
            ),
            (
                {"data": {"names": [{"name": "Muhammad Ben Abdul Aziz Ben Ali Bouyehia"}]}},
                {"data": {"names": [{"name": "Muhammad Ben Farim"}]}},
                False,
            ),
        ],
    )
    def test_extracts_country_codes(self, raw_sanction1, raw_sanction2, same_fathers_expected):
        s1, s2 = Sanction("", raw_sanction1, "person"), Sanction("", raw_sanction2, "person")
        v = FathersNamesFromAlias()
        s1.accept_visitor(v), s2.accept_visitor(v)
        f = SanctionFeatures.FATHER_PHONETIC
        same_fathers = s1.extracted_entities[f] == s2.extracted_entities[f]
        assert same_fathers == same_fathers_expected
