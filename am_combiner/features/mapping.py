from am_combiner.features.remote import Neo4jEmbeddingVisitor

from am_combiner.features.common import (
    EntityNameRemoverVisitor,
    SpacyArticleVisitor,
    FieldCleaningVisitor,
    TextCleaningVisitor,
    SanctionAliasExtractor,
    SanctionPrimariesExtractor,
    SanctionBirthExtractor,
    SanctionPassportVisitor,
    NationalityVisitor,
    CountryCodeVisitor,
    AmlTypeVisitor,
    FathersNamesFromAlias,
)
from am_combiner.features.date import DateStandardisationVisitor
from am_combiner.features.entity_linking import EnitityLinkVisitor
from am_combiner.features.geography import ArticleGeoVisitor
from am_combiner.features.profession import ProfessionVisitor
from am_combiner.features.sanction_term import SanctionTermVisitor, SanctionTermSpacyVisitor
from am_combiner.features.terms import ArticleKeywordVisitor, ArticleTermVisitor
from am_combiner.features.time import TimeStandardisationVisitor
from am_combiner.features.metadata_search import MetaKeyVisitor
from am_combiner.features.vectorisation import (
    FullArticleTextVectoriser,
    FullArticleFeaturesTextExtractor,
    FullArticleFeaturesTextVectoriser,
    BertVectoriser,
    TFIDFFullTextVisitorS3,
    JsonSummarizer,
    FullSanctionTextVectoriser,
)
from am_combiner.features.text_selector import ArticleSelectedTextVisitor
from am_combiner.features.graph_data import GraphDataVisitor
from am_combiner.features.domain import UrlDomainVisitor
from am_combiner.features.topic_model.topic_model import TopicVisitor
from am_combiner.features.topic_model.topic_tfidf_concat import TopicTfidfConcatVisitor
from am_combiner.features.organisation_visitors import (
    AddressVisitor,
    OrganisationIdentifiersVisitor,
    OrganisationAliasVisitor,
)

VISITORS_CLASS_MAPPING = {
    "ArticleKeywordVisitor": ArticleKeywordVisitor,
    "MetaKeyVisitor": MetaKeyVisitor,
    "ArticleTermVisitor": ArticleTermVisitor,
    "TimeStandardisationVisitor": TimeStandardisationVisitor,
    "DateStandardisationVisitor": DateStandardisationVisitor,
    "ProfessionVisitor": ProfessionVisitor,
    "TFIDFFullTextVisitor": FullArticleTextVectoriser,
    "EntityLinkVisitor": EnitityLinkVisitor,
    "FullArticleFeaturesTextExtractor": FullArticleFeaturesTextExtractor,
    "FullArticleFeaturesTextVectoriser": FullArticleFeaturesTextVectoriser,
    "EntityNameRemoverVisitor": EntityNameRemoverVisitor,
    "SpacyArticleVisitor": SpacyArticleVisitor,
    "UrlDomainVisitor": UrlDomainVisitor,
    "FieldCleaningVisitor": FieldCleaningVisitor,
    "TextCleaningVisitor": TextCleaningVisitor,
    "BertVisitor": BertVectoriser,
    "ArticleSelectedTextVisitor": ArticleSelectedTextVisitor,
    "GraphDataVisitor": GraphDataVisitor,
    "ArticleGeoVisitor": ArticleGeoVisitor,
    "TFIDFFullTextVisitorS3": TFIDFFullTextVisitorS3,
    "TFIDFFullTextVisitorS3_8000": TFIDFFullTextVisitorS3,
    "TFIDFFullTextVisitorS3_12000": TFIDFFullTextVisitorS3,
    "TopicVisitor": TopicVisitor,
    "TopicTfidfConcatVisitor": TopicTfidfConcatVisitor,
    "SanctionAliasExtractor": SanctionAliasExtractor,
    "SanctionPrimariesExtractor": SanctionPrimariesExtractor,
    "SanctionBirthExtractor": SanctionBirthExtractor,
    "SanctionPassportVisitor": SanctionPassportVisitor,
    "NationalityVisitor": NationalityVisitor,
    "SanctionTermVisitor": SanctionTermVisitor,
    "SanctionTermSpacyVisitor": SanctionTermSpacyVisitor,
    "CountryCodeVisitor": CountryCodeVisitor,
    "AmlTypeVisitor": AmlTypeVisitor,
    "JsonSummarizer": JsonSummarizer,
    "FullSanctionTextVectoriser": FullSanctionTextVectoriser,
    "Neo4jEmbeddingVisitor": Neo4jEmbeddingVisitor,
    "FathersNamesFromAlias": FathersNamesFromAlias,
    "AddressVisitor": AddressVisitor,
    "OrganisationIdentifiersVisitor": OrganisationIdentifiersVisitor,
    "OrganisationAliasVisitor": OrganisationAliasVisitor,
}
