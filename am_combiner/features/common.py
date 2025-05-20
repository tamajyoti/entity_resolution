import abc
from collections import defaultdict
from functools import reduce
from itertools import combinations
from typing import Union, Set, List
from unidecode import unidecode
import jellyfish

import regex as re
import spacy

from am_combiner.features.article import Article, Features
from am_combiner.features.sanction import Sanction, SanctionFeatures

MAX_ARTICLE_LEN = 999999


class ArticleVisitor(abc.ABC):

    """
    An abstract class representing a generic visitor.

    Classes that inherit from it are supposed to be feature extractors which may change the state of
    the article object.

    Methods
    -------
    visit_article(article: Article):
        Concrete implementations of this method will be changing the state of an Article object.

    """

    @abc.abstractmethod
    def visit_article(self, article: Article) -> None:
        """
        Visit the article and apply the changes.

        Parameters
        ----------
        article:
            The article to be visited.

        """
        pass


class SanctionVisitor(abc.ABC):

    """An abstract class representing a generic visitor for sanction object."""

    @abc.abstractmethod
    def visit_sanction(self, sanction: Sanction) -> None:
        """
        Visit the sanction and apply the changes.

        Parameters
        ----------
        sanction:
            The sanction to be visited.

        """
        pass


class SpacyArticleVisitor(ArticleVisitor):

    """
    A concrete implementation of the ArticleVisitor class.

    Extracts spacy tags from an article.

    """

    def __init__(
        self,
        do_coref_resolution: bool = False,
        lower_case_features: bool = False,
    ):
        super().__init__()
        self.do_coref_resolution = do_coref_resolution
        self.lower_case_features = lower_case_features

        self.nlp = spacy.load("en_core_web_sm")
        if self.do_coref_resolution:
            import neuralcoref

            neuralcoref.add_to_pipe(self.nlp)

    def visit_article(self, article: Article) -> None:
        """
        Visit the article and apply the tag extraction.

        Parameters
        ----------
        article:
            The article to be visited.

        """
        article_text = article.extracted_entities[Features.ARTICLE_TEXT]

        if len(article_text) > MAX_ARTICLE_LEN:
            print(f"Warning: spacy input truncated from {len(article_text)} to {MAX_ARTICLE_LEN}")
            article_text = article_text[:MAX_ARTICLE_LEN]

        spacy_tags = self.nlp(article_text)
        for extracted_entity in spacy_tags.ents:
            feature_ref = Features[extracted_entity.label_]
            feature_text = extracted_entity.text.strip()
            if self.lower_case_features:
                feature_text = feature_text.lower()

            article.extracted_entities[feature_ref].add(feature_text)

        article_sentences = [sent.text for sent in spacy_tags.sents]
        article.extracted_entities[Features.ARTICLE_SENTENCES] = article_sentences

        if self.do_coref_resolution:
            coreference_resolved_text = spacy_tags._.coref_resolved.replace("\n", "")
            # replace is used to remove new line added by coref resolution function
            coreference_resolved_clusters = spacy_tags._.coref_clusters
            article.extracted_entities[
                Features.COREFERENCE_RESOLVED_TEXT
            ] = coreference_resolved_text
            article.extracted_entities[
                Features.COREFERENCE_RESOLVED_CLUSTERS
            ] = coreference_resolved_clusters

            coreference_resolved_sentences = [
                sent.strip()
                for sent in coreference_resolved_text.split(".")
                if len(sent.strip()) > 0
            ]
            # cleaned the text file from new line issues
            article.extracted_entities[
                Features.COREFERENCE_RESOLVED_SENTENCES
            ] = coreference_resolved_sentences


class FieldCleaningVisitor(ArticleVisitor):

    """
    A concrete implementation of a visitor pattern.

    This implementation makes sure that any tokens in entity name are not part of extracted
    PERSON/ORG tuples.

    Attributes
    ----------
        feature_name: str
            The name of the field that should be cleaned from the entity name tokens.

    """

    def __init__(self, feature_name: Features, target_feature: Features):
        super().__init__()
        self.feature_name = feature_name
        self.target_feature = target_feature

    def visit_article(self, article: Article) -> None:
        """
        Make sure that a given field is cleaned from any tokens making an Article.entity_names.

        This is done to reduce the overall overcombination rate.

        Parameters
        ----------
        article:
            An article to be changed.

        """
        if self.feature_name in article.extracted_entities:
            entities = article.extracted_entities[self.feature_name]
            entities = remove_self_reference(entities, article.entity_name)
            article.extracted_entities[self.target_feature] = entities


class TextCleaningVisitor(ArticleVisitor):

    """Implements the text cleaner class to clean the article text of HTML tags."""

    def __init__(self):
        super().__init__()

    def visit_article(self, article: Article) -> None:
        """
        Clean the article text from HTML tags.

        Parameters
        ----------
        article:
            Article to be cleaned.

        """
        no_tags_text = TextCleaner.clean_html_tag(article.extracted_entities[Features.ARTICLE_TEXT])
        article.extracted_entities[Features.ARTICLE_TEXT] = no_tags_text


class EntityNameRemoverVisitor(ArticleVisitor):

    """A concrete implementation of the article visitor class."""

    def __init__(
        self,
        source_feature: Union[Features, str] = Features.ARTICLE_TEXT,
    ):
        super().__init__()
        self.source_feature = source_feature

    def visit_article(self, article: Article) -> None:
        """
        Visit the article and apply the name removal.

        Parameters
        ----------
        article:
            The article to be visited.

        """
        # TODO substring removal is not very good.
        #  E.g. John Smith vs Robert Johnson will lead to "  son"
        if not article.extracted_entities[Features.PERSON]:
            return
        # Split each name into tokens and create a set of unique tokens
        # TODO there has to be a better way to delete those names
        values = list(article.extracted_entities[Features.PERSON]) + [article.entity_name]
        names_bits = set(reduce(lambda x, y: x + y, [n.split() for n in values]))
        # Every single token from above is removed from the text
        # the set is sorted since the iteration order is not guaranteed for sets and that
        # means that elements can be removed in a different order from one run to another.
        # The resulting string depends on the order substrings are removed from it.
        # In order to get a reproducible set of results, we fix the order to avoid randomness.
        for bit in sorted(names_bits):
            article.extracted_entities[self.source_feature] = article.extracted_entities[
                self.source_feature
            ].replace(bit, "")


def remove_self_reference(field_values: Union[List, Set], entity_name: str) -> Union[List, Set]:
    """
    Take a list of spacy tags and make sure that entity name is not presented in any of those.

    In this context "not presented" means no name string tokens appear in the extracted entity.

    Having those tokens there dramatically increases the chances of overcombination,
    that is why we remove them.

    Parameters
    ----------
    field_values:
        List or set of extracted spacy tuples.
    entity_name:
        An entity name to be removed.

    Returns
    -------
        Cleaned up extracted tags

    """
    if not isinstance(field_values, (list, set)):
        raise ValueError("Only set of lists are allowed here")
    cls = field_values.__class__
    out = []
    name_tokens = entity_name.lower().split()
    for value in field_values:
        ignore = False
        for token in name_tokens:

            # Do not match on middle name single letter:
            if (len(token) == 2) and token.endswith(r"."):
                continue
            token = token.replace(".", r"\.")
            pattern = re.compile(r"\b" + token + r"\b", re.IGNORECASE)
            if re.search(pattern, value) is not None:
                ignore = True
                break
        if ignore:
            continue
        out.append(value)
    return cls(out)


def convert_name_to_keyword_tokens(name: str) -> List[str]:
    """Convert name into keyword."""
    name = unidecode(name)
    name = name.lower()

    name_tokens = re.sub(r"(,|/|\)|\(|\.|-|;|`|')", " ", name).split(" ")
    name_tokens = [name_token for name_token in name_tokens if len(name_token) > 2]

    if name_tokens:
        return sorted(name_tokens)
    return [name]


def get_phonetic_keyword(tokenized_name: List[str]) -> List[str]:
    """Convert each token into phonetic encoding."""
    phonetic_tokens = [jellyfish.soundex(name_token) for name_token in tokenized_name]
    return sorted(phonetic_tokens)


# Cell
class TextCleaner:

    """
    A class which contains a library of functions to clean a text.

    All text cleaning functions are to be contained in the class.

    ** In future we can add more functions to clean the text using more functions.

    Methods
    -------
    clean_tag(text: str):
        Accept a text string containing an article which needs to be cleaned.

    """

    @staticmethod
    def clean_html_tag(text: str) -> str:
        """
        Remove HTML tags from the article text.

        Parameters
        ----------
        text:
            The article text to be cleaned.

        Returns
        -------
            Cleaned text.

        """
        return re.sub("<.*?>", "", text)


class SanctionAliasExtractor(SanctionVisitor):

    """Extracts all aliases in the sanction."""

    def __init__(self):
        super().__init__()

    def _get_names(self, records) -> Set[str]:
        names = []
        for record in records:
            if "name" in record:
                names.append(record["name"])
        return set(names)

    @staticmethod
    def _get_bigrams(aliases_tokens: List[List[str]]) -> List[str]:
        bigrams = []
        for alias_tokens in aliases_tokens:
            combs = combinations(alias_tokens, 2)
            bigrams += ["+".join(t) for t in combs]
        return set(bigrams)

    @staticmethod
    def _check_fields_exist(ent) -> bool:
        if "data" not in ent:
            return False
        if "names" not in ent["data"]:
            return False
        return True

    def visit_sanction(self, sanction: Sanction) -> None:
        """Extract aliases and primary names."""
        if not self._check_fields_exist(sanction.raw_entity):
            return

        aliases = self._get_names(sanction.raw_entity["data"]["names"])
        kw_toks = [convert_name_to_keyword_tokens(alias) for alias in aliases]
        ph_toks = [get_phonetic_keyword(alias_tokens) for alias_tokens in kw_toks]

        sanction.extracted_entities[SanctionFeatures.ALIAS] = aliases
        sanction.extracted_entities[SanctionFeatures.ALIAS_KEYWORD] = set(
            ["+".join(toks) for toks in kw_toks]
        )
        sanction.extracted_entities[SanctionFeatures.ALIAS_PHONETIC] = set(
            ["+".join(toks) for toks in ph_toks]
        )
        sanction.extracted_entities[SanctionFeatures.ALIAS_KEYWORD_BIGRAMS] = self._get_bigrams(
            kw_toks
        )
        sanction.extracted_entities[SanctionFeatures.ALIAS_PHONETIC_BIGRAMS] = self._get_bigrams(
            ph_toks
        )


class SanctionPrimariesExtractor(SanctionAliasExtractor):

    """Extracts all primaries in the sanction."""

    def __init__(self):
        super().__init__()

    def _get_names(self, records) -> Set[str]:
        primary = []
        for record in records:
            if "name_type" in record and record["name_type"] == "primary":
                if "name" in record:
                    primary.append(record["name"])
        return set(primary)

    def visit_sanction(self, sanction: Sanction) -> None:
        """Extract aliases and primary names."""
        if not self._check_fields_exist(sanction.raw_entity):
            return

        primaries = self._get_names(sanction.raw_entity["data"]["names"])
        kw_toks = [convert_name_to_keyword_tokens(alias) for alias in primaries]

        sanction.extracted_entities[SanctionFeatures.PRIMARY] = primaries
        sanction.extracted_entities[SanctionFeatures.PRIMARY_KEYWORD] = [
            "+".join(toks) for toks in kw_toks
        ]


class SanctionBirthExtractor(SanctionVisitor):

    """Extracts all DOBs and YOB's for primaries."""

    def __init__(
        self,
        feature_name_yob: SanctionFeatures = SanctionFeatures.YOB,
        feature_name_dob: SanctionFeatures = SanctionFeatures.DOB,
        feature_name_yob_known: SanctionFeatures = SanctionFeatures.YOB_KNOWN,
    ):
        super().__init__()
        self.feature_yob = feature_name_yob
        self.feature_dob = feature_name_dob
        self.feature_yob_known = feature_name_yob_known

    @staticmethod
    def _get_birth_years(records) -> Set[str]:
        """Extract year of birth range from primaries."""
        years = []
        for record in records:
            if "min_date" in record:
                year_str = str(record["min_date"])[:4]
                if year_str.isdigit():
                    years.append(int(year_str))
        if years:
            return set(range(min(years), max(years) + 1))
        else:
            return set()

    @staticmethod
    def _get_known_birth_years(records) -> Set[str]:
        """Extract DOBs for primaries from the sanction data."""
        dobs = set()
        for record in records:
            if "min_date" in record and "max_date" in record:
                year_min, month_min, day_min = record["min_date"].split("-")
                year_max, month_max, day_max = record["max_date"].split("-")
                if (day_min, month_min, day_max, month_max) in {
                    ("01", "01", "12", "31"),
                    ("01", "01", "31", "12"),
                }:
                    # Only YOB is known
                    dobs.add(year_min)
                elif (day_min, month_min, day_max, month_max) == ("01", "01", "01", "01"):
                    dobs.add(year_min)
                    dobs.add(year_max)
                else:
                    dobs.add(year_min)
        return dobs

    @staticmethod
    def _get_birth_dates(records) -> Set[str]:
        """Extract DOBs for primaries from the sanction data."""
        dobs = []
        for record in records:
            if "min_date" in record and "max_date" in record:
                if record["min_date"] == record["max_date"]:
                    birth_date = record["min_date"]
                    dobs.append(birth_date)
        if dobs:
            return set(dobs)
        else:
            return set()

    def visit_sanction(self, sanction: Sanction) -> None:
        """Extract aliases and primary names."""
        if "data" not in sanction.raw_entity:
            return
        if "births" not in sanction.raw_entity["data"]:
            return

        births = sanction.raw_entity["data"]["births"]
        if births is None:
            return

        sanction.extracted_entities[self.feature_yob] = self._get_birth_years(births)
        sanction.extracted_entities[self.feature_yob_known] = self._get_known_birth_years(births)
        sanction.extracted_entities[self.feature_dob] = self._get_birth_dates(births)


class CountryCodeVisitor(SanctionVisitor):

    """Extracts all country codes."""

    def __init__(self, remove_internationals: bool = True):
        super().__init__()
        self.remove_internationals = remove_internationals

    def visit_sanction(self, entity: Sanction) -> None:
        """Extract country codes."""
        if "data" not in entity.raw_entity or "locations" not in entity.raw_entity["data"]:
            return
        if entity.raw_entity["data"]["locations"] is None:
            return

        locations = entity.raw_entity["data"]["locations"]

        # remove internationals:
        if self.remove_internationals:
            for location in locations:
                if "original_name" in location:
                    if location["original_name"] == "International":
                        return

        ccs = [
            location["country_code"]
            for location in locations
            if "country_code" in location and location["country_code"] is not None
        ]
        entity.extracted_entities[SanctionFeatures.COUNTRY_CODE] = set(ccs)


class SanctionPassportVisitor(SanctionVisitor):

    """Responsible for useful passport information extraction."""

    feature_mapping = {
        "DATE": SanctionFeatures.PASSPORT_DATES,
        "GPE": SanctionFeatures.PASSPORT_GPE,
        "NORP": SanctionFeatures.PASSPORT_NORPS,
    }

    def __init__(self):
        super().__init__()
        self.nlp = spacy.load("en_core_web_sm")

    @staticmethod
    def has_nums(t):
        """Check if the string contains any numbers."""
        for c in t:
            if c.isdigit():
                return True
        return False

    @staticmethod
    def _extract_pid(k):
        """Extract passport id from passport string."""
        k = " ".join(re.split(";|,|-|/|\(|\)", k))
        tokens = k.split()
        tokens = [
            t for t in tokens if t.isalnum() and len(t) > 4 and SanctionPassportVisitor.has_nums(t)
        ]
        tokens = "+".join(sorted(tokens)).lower()

        return tokens

    def visit_sanction(self, sanction: Sanction) -> None:
        """Extract passport information from a sanction record."""
        entity = sanction.raw_entity
        if "data" not in entity or "passports" not in entity["data"]:
            return
        passports = entity["data"]["passports"]
        if not passports:
            return
        passports_info = defaultdict(set)
        sanction.extracted_entities[SanctionFeatures.PASSPORT_RAW] = [
            p["passport"] for p in passports
        ]
        for passport in passports:
            passport_ = passport["passport"]
            pid = SanctionPassportVisitor._extract_pid(passport_)
            if pid:
                sanction.extracted_entities[SanctionFeatures.PASSPORT_ID].add(pid)
            tags = self.nlp(passport_)
            for ent in tags.ents:
                feature = SanctionPassportVisitor.feature_mapping.get(ent.label_)
                if not feature:
                    continue
                passports_info[feature].add(ent.text.strip().lower())
        sanction.extracted_entities.update(passports_info)


class NationalityVisitor(SanctionVisitor):

    """Sanction nationality extractor."""

    def visit_sanction(self, sanction: Sanction) -> None:
        """Concrete implementation of the visitor.Extract a list of nationalities."""
        entity = sanction.raw_entity
        if "data" not in entity or "nationalities" not in entity["data"]:
            return
        nationalities = entity["data"]["nationalities"]
        if not nationalities:
            return
        for nationality in nationalities:
            sanction.extracted_entities[SanctionFeatures.NATIONALITIES].add(
                nationality["country_code"]
            )


class AmlTypeVisitor(SanctionVisitor):

    """Extracts all aml types."""

    def __init__(self):
        super().__init__()

    def visit_sanction(self, entity: Sanction) -> None:
        """Extract aml types."""
        if "data" not in entity.raw_entity or "aml_types" not in entity.raw_entity["data"]:
            return
        if entity.raw_entity["data"]["aml_types"] is None:
            return

        aml_types = entity.raw_entity["data"]["aml_types"]

        amls = [
            aml["aml_type"]
            for aml in aml_types
            if "aml_type" in aml and aml["aml_type"] is not None
        ]
        entity.extracted_entities[SanctionFeatures.AML_TYPES] = set(amls)


class FathersNamesFromAlias(SanctionAliasExtractor):

    """Extract fathers phonetics from aliases."""

    def __init__(self):
        super().__init__()
        self.FATHERS_DENOMINATIONS = ["ben", "s/o"]
        self.min_tokens = 3

    def _get_fathers_phonetics(self, aliases: str) -> Set[str]:
        """Get a set of phonetic representations of paternal ancestors."""
        fathers_phonetics = set()
        for alias in aliases:
            alias_toks = alias.lower().split(" ")
            for fd in self.FATHERS_DENOMINATIONS:
                if fd in alias_toks[:-1]:
                    inx = alias_toks.index(fd)
                    father_alias = " ".join(alias_toks[(inx + 1) :])
                    father_kw_toks = convert_name_to_keyword_tokens(father_alias)
                    father_phonetic = "+".join(get_phonetic_keyword(father_kw_toks))
                    fathers_phonetics.add(father_phonetic)
        return fathers_phonetics

    def visit_sanction(self, sanction: Sanction) -> None:
        """Extract phonetic representations for paternal alias."""
        if "data" not in sanction.raw_entity:
            return
        if "names" not in sanction.raw_entity["data"]:
            return

        aliases = self._get_names(sanction.raw_entity["data"]["names"])
        sanction.extracted_entities[SanctionFeatures.FATHER_PHONETIC] = self._get_fathers_phonetics(
            aliases
        )
