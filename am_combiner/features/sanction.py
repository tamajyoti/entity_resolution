from enum import Enum, auto
from collections import defaultdict


class SanctionFeatures(Enum):

    """Define the list of possible sanction features."""

    SANCTION_TYPE = auto()
    PRIMARY = auto()
    PRIMARY_KEYWORD = auto()
    ALIAS = auto()
    ALIAS_KEYWORD = auto()
    ALIAS_PHONETIC = auto()
    ALIAS_KEYWORD_BIGRAMS = auto()
    ALIAS_PHONETIC_BIGRAMS = auto()
    ADDRESS_COUNTRY = auto()
    ADDRESS_TOKENS = auto()
    YOB = auto()
    DOB = auto()
    YOB_KNOWN = auto()
    PASSPORT_RAW = auto()
    PASSPORT_ID = auto()
    PASSPORT_GPE = auto()
    PASSPORT_DATES = auto()
    PASSPORT_NORPS = auto()
    NATIONALITIES = auto()
    IDENTIFICATION_NUMBER = auto()
    AML_TYPES = auto()
    FATHER_PHONETIC = auto()
    # List of display fields
    DESIGNATION_DATE = auto()
    DESIGNATION_YEAR = auto()
    PROGRAM = auto()
    REASON = auto()
    RELATED_URL = auto()
    FUNCTION = auto()
    ADDRESS = auto()
    LISTING_ID = auto()
    LEGAL_BASIS = auto()
    REGIME = auto()
    ORG_IDS = auto()
    ISSUING_AUTHORITY = auto()
    TITLE = auto()
    OTHER_INFORMATION = auto()
    PERSON = auto()
    FAC = auto()
    ORG = auto()
    CARDINAL = auto()
    COUNTRY_CODE = auto()
    NORP = auto()
    EVENT = auto()
    DATE = auto()
    GPE = auto()
    TIME = auto()
    TIME_CLEAN = auto()
    LOC = auto()
    WORK_OF_ART = auto()
    LAW = auto()
    PERCENT = auto()
    PRODUCT = auto()
    ORDINAL = auto()
    MONEY = auto()
    QUANTITY = auto()
    LANGUAGE = auto()
    TERM = auto()
    TERM_KEYWORD = auto()
    TFIDF = auto()
    FULL_TEXT = auto()
    CHAMBER = auto()
    POLITICAL_PARTY = auto()
    POLITICAL_REGION = auto()
    REMOTE = auto()  # features fetched from the db


class Sanction:

    """
    A container that represents an sanction and a structured profile id associated with it.

    Upon an object creation, it will run spaCy extractors and extract all available spacy tags.

    This class utilises the Visitor pattern and is able to accept visitors that are able
    to change the class state. This approach allows one to implement new feature extractors
    without affecting anyone else.

    Attributes
    ----------
    raw_entity: str
        raw entity data from structured profile
    sanction_id: str
        structured profile ID in format "SM:S:..."
    extracted_entities: defaultdict(set)
        Extracted entities.

    Methods
    -------
    accept_visitor(visitor: Visitor):
        Accept a visitor and make it visit the sanction.
        This has a potential to change the object's state.

    """

    def __init__(self, sanction_id: str, raw_entity, sanction_type: str):
        """
        Wrap sanction id and corresponding raw entity data together.

        Parameters
        ----------
        sanction_id:
            sanction id string of format "SM:S:..."
        raw_entity:
            json from mongo DB containing entity data.
        sanction_type:
            type of sanction is used: person, organisation, vessels, undefined...

        """
        self.raw_entity = raw_entity
        self.sanction_id = sanction_id
        self.extracted_entities = defaultdict(set)
        self.type = sanction_type

    def accept_visitor(self, visitor) -> None:
        """
        Accept a visitor and make it visit the sanction.

        Parameters
        ----------
        visitor:
            An implementation of SanctionVisitor.

        """
        visitor.visit_sanction(self)
