import re
from typing import Set, List, Dict, Optional, Union
from unidecode import unidecode

from am_combiner.features.sanction import Sanction, SanctionFeatures
from am_combiner.features.geography import get_full_geo_resolver
from deep_translator import GoogleTranslator
from am_combiner.features.common import (
    SanctionVisitor,
    SanctionAliasExtractor,
)


SPLIT_PATTERN = r"(,|/|\)|\(|\.|-|;|`|')"


def check_for_field(json, field: str) -> bool:
    """Check if field exists and maps to list."""
    if field in json:
        if isinstance(json[field], list):
            return True
    return False


class OrganisationIdentifiersVisitor(SanctionVisitor):

    """Extracts all aliases in the sanction."""

    EXCEPTION_CODE = "1001"

    @staticmethod
    def _verify_codes(tok: str) -> Optional[str]:
        """Extract id codes from Registration ID value."""
        if len(tok) >= 5 and tok[2:].isnumeric():
            if not tok.endswith(OrganisationIdentifiersVisitor.EXCEPTION_CODE):
                return tok

    @staticmethod
    def _verify_year(tok: str) -> Optional[int]:
        """Extract year from Designation Date value string."""
        if len(tok) == 4 and tok.isnumeric():
            return int(tok)

    @staticmethod
    def _extract_identifiers(ent, title: str, func) -> Set[Union[str, int]]:
        """Search for Designation date json."""
        out = set()
        if check_for_field(ent, "display_fields"):
            for display_field in ent["display_fields"]:
                if display_field["title"] == title:
                    if "value" in display_field:
                        toks = re.sub(SPLIT_PATTERN, " ", display_field["value"]).split(" ")
                        for tok in toks:
                            out.add(func(tok))

        return set([o for o in out if o])

    def visit_sanction(self, sanction: Sanction) -> None:
        """Extract organisation specific identifiers."""
        if "data" not in sanction.raw_entity:
            return

        sanction.extracted_entities[SanctionFeatures.ORG_IDS] = self._extract_identifiers(
            sanction.raw_entity["data"], "Registration Number", self._verify_codes
        )
        sanction.extracted_entities[SanctionFeatures.DESIGNATION_YEAR] = self._extract_identifiers(
            sanction.raw_entity["data"], "Designation Date", self._verify_year
        )


class OrganisationAliasVisitor(SanctionAliasExtractor):

    """Extracts all aliases in the sanction."""

    ORG_ALIAS_STOPWORDS = {"ltd", "llc", "lcc", "inc", "corp", "corporation", "company"}

    @staticmethod
    def _verify_token(tok: str) -> bool:
        if tok.isnumeric():
            return True
        if len(tok) > 2:
            if tok not in OrganisationAliasVisitor.ORG_ALIAS_STOPWORDS:
                return True
        return False

    def _convert_to_keyword_tokens(self, name: str) -> List[str]:
        """Convert org alias into keyword tokens."""
        out = []
        name = unidecode(name)
        name = name.lower()

        # deal with in-brackets values
        # to match "Big Fat Shop [BFS]" with "Big Fat Shop" and "Big Fat Shop BFS".
        name = re.sub(r"({|\[)", "(", name)
        name = re.sub(r"(}|])", ")", name)
        if "(" in name and ")" in name.split("(")[1]:
            str_inx = name.index("(")
            end_inx = str_inx + name.split("(")[1].index(")") + 2
            name_without_brackets = name[:str_inx] + name[end_inx:]
            out += self._convert_to_keyword_tokens(name_without_brackets)

        # Split on aka
        # to match "First a.k.a. Second" with "First" and "Second".
        if "a.k.a." in name:
            inx = name.index("a.k.a.")
            out += self._convert_to_keyword_tokens(name[:inx])
            out += self._convert_to_keyword_tokens(name[(inx + 6) :])

        name = re.sub(r"&", " and ", name)
        name_tokens = re.sub(SPLIT_PATTERN, " ", name).split(" ")
        name_tokens = [tok for tok in name_tokens if self._verify_token(tok)]

        if name_tokens:
            out.append("+".join(sorted(name_tokens)))
        return out

    def visit_sanction(self, sanction: Sanction) -> None:
        """Extract organisation specific identifiers."""
        if not self._check_fields_exist(sanction.raw_entity):
            return

        aliases = self._get_names(sanction.raw_entity["data"]["names"])
        sanction.extracted_entities[SanctionFeatures.ALIAS] = set(aliases)
        aliases_k = []
        for alias in aliases:
            aliases_k += self._convert_to_keyword_tokens(alias)
        sanction.extracted_entities[SanctionFeatures.ALIAS_KEYWORD] = set(aliases_k)


class AddressVisitor(SanctionVisitor):

    """Extracts all addresses in the sanction. Intended for organisations type."""

    ADDRESS_STOPWORDS = {
        "city",
        "country",
        "street",
        "road",
        "avenue",
        "town",
        "junction",
        "alley",
        "business",
        "village",
        "box",
        "mailbox",
        "post",
        "square",
        "floor",
        "building",
        "former",
        "center",
        "centre",
        "central",
        "park",
        "branch",
        "house",
        "area",
        "apartment",
        "flat",
        "lane",
        "coast",
        "suite",
        "region",
        "district",
        "company",
        "registration",
        "federation",
        "republic",
        "kingdom",
        "democratic",
        "highway",
        "room",
        "office",
        "block",
        "number",
        "industrial",
        "united",
        "island",
        "islands",
        "isles",
        "code",
    }
    MAX_CHAR = 4999

    def __init__(self):
        self.GeoResolver = get_full_geo_resolver()
        self.Translator = GoogleTranslator(source="auto", target="en")

    @staticmethod
    def _get_address_text(address_jsons: List[Dict[str, str]]) -> Set[str]:
        out = set()
        for address_json in address_jsons:
            for field in ["original_address", "value", "name"]:
                if field in address_json:
                    out.add(address_json[field])

        if None in out:
            out.remove(None)
        return out

    @staticmethod
    def _get_address_jsons(ent) -> List[Dict[str, str]]:
        out = []

        for field in ["addresses", "locations"]:
            if check_for_field(ent, field):
                for address in ent[field]:
                    out.append(address)

        if check_for_field(ent, "display_fields"):
            for display_field in ent["display_fields"]:
                if display_field["title"].lower() == "address":
                    out.append(display_field)

        out = [o for o in out if o]
        return out

    @staticmethod
    def _tokenize_addresses(addresses: Set[str]) -> Set[str]:
        toks = []
        for address in addresses:
            address_toks = re.sub(SPLIT_PATTERN, " ", address).split(" ")
            toks += address_toks

        toks = [t.lower() for t in toks if (t.isnumeric() or len(t) > 3)]
        toks = [t for t in toks if t not in AddressVisitor.ADDRESS_STOPWORDS]
        return set(toks)

    def _translate_addresses(self, addresses: Set[str]) -> Set[str]:
        eng_addresses = []
        for address in addresses:
            if unidecode(address) != address:
                address_to_translate = address[: AddressVisitor.MAX_CHAR]
                try:
                    eng_address = self.Translator.translate(address_to_translate)
                except ValueError:
                    eng_address = address
            else:
                eng_address = address
            eng_addresses.append(eng_address)
        return set(eng_addresses)

    def _extract_countries(self, address_jsons, english_addresses: Set[str]) -> Set[str]:
        countries = []
        for address_json in address_jsons:
            if "country" in address_json and isinstance(address_json["country"], str):
                country = address_json["country"].strip()
                resolved_country = self.GeoResolver.resolve_geo_name(country, {"final": True})
                countries += resolved_country

        for add in english_addresses:
            address_toks = re.sub(r"(,|\.|:)", ";", add).split(";")
            for address_tok in address_toks:
                resolved_country = self.GeoResolver.resolve_geo_name(
                    address_tok.strip(), {"final": True}
                )
                countries += resolved_country
        return set(countries)

    def visit_sanction(self, sanction: Sanction) -> None:
        """Extract aliases and primary names."""
        if "data" not in sanction.raw_entity:
            return

        address_jsons = self._get_address_jsons(sanction.raw_entity["data"])
        addresses = self._get_address_text(address_jsons)
        if not addresses:
            return

        english_addresses = self._translate_addresses(addresses)
        address_toks = self._tokenize_addresses(english_addresses)
        address_country = self._extract_countries(address_jsons, english_addresses)

        sanction.extracted_entities[SanctionFeatures.ADDRESS_TOKENS] = address_toks
        sanction.extracted_entities[SanctionFeatures.ADDRESS] = english_addresses
        sanction.extracted_entities[SanctionFeatures.ADDRESS_COUNTRY] = address_country
