import pytest

from am_combiner.features.organisation_visitors import (
    check_for_field,
    OrganisationIdentifiersVisitor,
    OrganisationAliasVisitor,
    AddressVisitor,
)


@pytest.mark.parametrize(
    ["input_json", "expected_output"],
    [
        ({"a": None}, False),
        ({"b": [1, 2]}, False),
        ({"a": [1, 2]}, True),
    ],
)
def test_check_for_field(input_json, expected_output):
    assert check_for_field(input_json, "a") == expected_output


@pytest.mark.parametrize(
    ["registration_input", "expected_output"],
    [("7610076500", "7610076500"), ("IR12751", "IR12751"), ("VALUABLE", None), ("52", None)],
)
def test_verify_codes(registration_input, expected_output):
    codes = OrganisationIdentifiersVisitor._verify_codes(registration_input)
    assert codes == expected_output


@pytest.mark.parametrize(
    ["designation_date", "expected_output"],
    [("1998", 1998), ("june", None)],
)
def test_verify_years(designation_date, expected_output):
    years = OrganisationIdentifiersVisitor._verify_year(designation_date)
    assert years == expected_output


@pytest.mark.parametrize(
    ["input_json", "expected_output"],
    [
        ({}, set()),
        ({"display_fields": [{"title": "a", "value": "b"}]}, set(["b"])),
        (
            {"display_fields": [{"title": "a", "value": "b"}, {"title": "a", "value": "c"}]},
            set(["b", "c"]),
        ),
        ({"display_fields": []}, set([])),
        ({"display_fields": [{"title": "a"}]}, set([])),
    ],
)
def test_extract_identifiers(input_json, expected_output):
    ids = OrganisationIdentifiersVisitor._extract_identifiers(input_json, "a", lambda x: x)
    assert ids == expected_output


@pytest.mark.parametrize(
    ["token", "expected_output"],
    [("13", True), ("co", False), ("company", False), ("gazprom", True), ("ikea", True)],
)
def test_verify_token(token, expected_output):
    output = OrganisationAliasVisitor._verify_token(token)
    assert output == expected_output


@pytest.mark.parametrize(
    ["inp1", "inp2", "is_connected"],
    [
        ("Central Bank of Syria", "CENTRAL BANK SYRIA (CBS)", True),
        ("Central Bank of Syria CBS", "CENTRAL BANK SYRIA (CBS)", True),
        ("Institute of Nuclear Research", "[Iran] Institute of Nuclear Research", True),
        (
            "Korean Workers Party a.k.a. Propaganda And Agitation Department",
            "Propaganda And Agitation Department",
            True,
        ),
        (
            "Korean Workers Party a.k.a. Propaganda And Agitation Department",
            "Korean Workers Party",
            True,
        ),
        ("Huawei Software Technologies Co., Ltd.", "Huawei Software Technologies", True),
        ("Ltd Kingly Won International Co.", "Kingly Won International Company", True),
        ("AA ENERGY FZC", "FZC", False),
        ("Korea Haegumgang Trading Corp.", "Korea Haegumgang Trading Corporation", True),
    ],
)
def test_convert_to_keyword_tokens(inp1, inp2, is_connected):
    visitor = OrganisationAliasVisitor()
    out1 = visitor._convert_to_keyword_tokens(name=inp1)
    out2 = visitor._convert_to_keyword_tokens(name=inp2)
    inter = set(out1).intersection(set(out2))
    assert (len(inter) > 0) == is_connected


@pytest.mark.parametrize(
    ["address_jsons", "expected_output"],
    [
        (
            [{"original_address": "Washington, US"}, {"name": "White House"}],
            ["Washington, US", "White House"],
        ),
        (
            [{"original_address": "Washington, US", "name": "White House"}],
            ["Washington, US", "White House"],
        ),
        ([{}], []),
    ],
)
def test_get_address_text(address_jsons, expected_output):
    output = AddressVisitor._get_address_text(address_jsons)
    assert set(output) == set(expected_output)


@pytest.mark.parametrize(
    ["address_jsons", "expected_output"],
    [
        ({"addresses": [{"value": "Washington, US"}]}, [{"value": "Washington, US"}]),
        ({"addresses": None}, []),
        ({"locations": [{"country": "US"}]}, [{"country": "US"}]),
        (
            {"display_fields": [{"country": "US", "title": "address"}]},
            [{"country": "US", "title": "address"}],
        ),
        ({"display_fields": [{"value": "general", "title": "function"}]}, []),
    ],
)
def test_get_address_jsons(address_jsons, expected_output):
    output = AddressVisitor._get_address_jsons(address_jsons)
    assert output == expected_output


@pytest.mark.parametrize(
    ["adresses", "expected_output"],
    [
        (["Erasmus Building", "Queen's college"], set(["college", "erasmus", "queen"])),
        (["201 et-Aliegri, Bagdad"], set(["aliegri", "bagdad", "201"])),
    ],
)
def test_tokenize_addresses(adresses, expected_output):
    output = AddressVisitor._tokenize_addresses(adresses)
    assert output == expected_output


@pytest.mark.parametrize(
    ["adresses", "expected_output"], [(["مجمع نووي"], set(["nuclear complex"]))]
)
def test_translate_addresses(adresses, expected_output):
    visitor = AddressVisitor()
    output = visitor._translate_addresses(adresses)
    assert output == expected_output


@pytest.mark.parametrize(
    ["jsons", "adresses", "expected_output"],
    [
        ([{}], ["nuclear complex, Bagdad, Iraq"], set(["iraq"])),
        (
            [{"country": "Iran"}],
            ["nuclear complex, Bagdad, Iraq"],
            set(["islamic republic of iran", "iraq"]),
        ),
        (
            [],
            ["country: North Korea; city: Pyongyang"],
            set(["democratic people's republic of korea"]),
        ),
    ],
)
def test_extract_countries(jsons, adresses, expected_output):
    visitor = AddressVisitor()
    output = visitor._extract_countries(jsons, adresses)
    assert output == expected_output
