from collections import defaultdict
from enum import Enum, auto
from typing import List, Tuple, Dict, Optional

import scipy
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span


class Features(Enum):

    """Define the list of possible features."""

    PERSON = auto()
    PERSON_CLEAN = auto()
    FAC = auto()
    ORG = auto()
    ORG_CLEAN = auto()
    CARDINAL = auto()
    NORP = auto()
    EVENT = auto()
    DATE = auto()
    DATE_CLEAN = auto()
    GPE = auto()
    GPE_CLEAN = auto()
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
    PROFESSION_NLTK_KEYWORD = auto()
    PROFESSION_KEYWORD_KEYWORD = auto()
    PROFESSION_DEPENDENCY_PARSING = auto()
    DBPEDIA_ENTITY_TUPLES = auto()
    DBPEDIA_ENTITY_URI = auto()
    TFIDF_FULL_TEXT = auto()
    TFIDF_FULL_TEXT_8000 = auto()
    TFIDF_FULL_TEXT_12000 = auto()
    TFIDF_SELECTED_TEXT = auto()
    FULL_TEXT_FEATURES = auto()
    TFIDF_FULL_TEXT_FEATURES = auto()
    BERT_FEATURES = auto()
    COREFERENCE_RESOLVED_TEXT = auto()
    COREFERENCE_RESOLVED_CLUSTERS = auto()
    COREFERENCE_RESOLVED_SENTENCES = auto()
    TFIDF_COREFERENCE_RESOLVED_TEXT = auto()
    ARTICLE_TEXT = auto()
    AM_CATEGORY = auto()
    DOMAIN = auto()

    # A list of strings containing the sentences of the article
    ARTICLE_SENTENCES = auto()

    # A subset of the article sentences. Initially, the sentences containing the entity name.
    ARTICLE_TEXT_SELECTED = auto()

    # Used for 3D web visualization
    WEB_GRAPH_VIS = auto()

    # DOB from a dataframe
    DOB = auto()

    # TOPIC from a text
    TOPIC_IDS = auto()
    TOPIC_DISTRIBUTION = auto()
    TFIDF_TOPIC_CONCAT = auto()


class Article:

    """
    A container that represents an article and a name associated with it.

    Upon an object creation, it will run spaCy extractors and extract all available spacy tags.

    This class utilises the Visitor pattern and is able to accept visitors that are able
    to change the class state. This approach allows one to implement new feature extractors
    without affecting anyone else.

    Attributes
    ----------
    entity_name: str
        Name of the entity an article is related to.
    article_text: str
        String representing article text. Just a normal raw text, nothing else.
    extracted_entities: defaultdict(set)
        Extracted entities.

    Methods
    -------
    accept_visitor(visitor: Visitor):
        Accept a visitor and make it visit the article.
        This has a potential to change the object's state.
    dump_tuples():
        Return a list of tuples of length 2 which represent a spaCy extracted object and its tag.

    """

    def __init__(
        self,
        entity_name: str,
        article_text: str,
        url: str = None,
        meta: Optional[Dict[str, str]] = None,
    ):
        """
        Wrap an article text and an entity name together.

        Upon creation immediately extracts spacy tags and stores them in the class.

        Parameters
        ----------
        entity_name:
            Name of the entity an article is related to.
        article_text:
            String representing article text. Just a normal raw text, nothing else.
        url:
            The url of the article.
        meta:
            The meta data keys to be fetched for more information

        """
        self.entity_name = entity_name
        self.article_text = article_text
        self.url = url
        self.meta = meta if meta else {}
        self.extracted_entities = defaultdict(set)
        self.extracted_entities[Features.ARTICLE_TEXT] = article_text

    def accept_visitor(self, visitor) -> None:
        """
        Accept a visitor and make it visit the article.

        Parameters
        ----------
        visitor:
            An implementation of ArticleVisitor.

        """
        visitor.visit_article(self)

    def dump_tuples(self) -> List[Tuple[str, str]]:
        """
        Dump pairs of extracted object and its tag.

        Returns
        -------
            A list of tuples (entity_type, entity_raw_text) that can be used further
            down the pipeline.

        """
        output = []
        ignored = {Features.ARTICLE_TEXT}
        for tag_type, entities in self.extracted_entities.items():
            if tag_type in ignored:
                continue
            if isinstance(entities, str):
                output.append((entities, tag_type))
            for entity in entities:
                if isinstance(entity, (Span, Doc)):
                    output.append((entity.text, tag_type))
                elif isinstance(entity, str):
                    output.append((entity, tag_type))
                elif isinstance(entity, scipy.sparse.csr.csr_matrix):
                    # We do not dump sparse matrices here
                    continue
                else:
                    raise ValueError(
                        "Unknown datatype in entity list. Don't know how to process it"
                    )
        return output
