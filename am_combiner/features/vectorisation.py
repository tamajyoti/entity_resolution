import pickle
from typing import Union, Dict

import numpy as np
from sentence_transformers import SentenceTransformer

from scipy.sparse import csr_matrix

from am_combiner.combiners.tfidf import get_features_from_article
from am_combiner.features.article import Article
from am_combiner.features.article import Features
from am_combiner.features.common import ArticleVisitor, SanctionVisitor
from am_combiner.features.sanction import Sanction, SanctionFeatures
from am_combiner.utils.replace import replace_entity_name
from am_combiner.utils.storage import ensure_s3_resource_exists


class FullArticleTextVectoriser(ArticleVisitor):

    """
    Vectorises the full article text.

    The vectorised form can be used by downstream ML components.

    Attributes
    ----------
    vectoriser: object
        A vectoriser given must implement transform method.
        The methods must take a list of articles and output a vector form of a given article.

    """

    def __init__(
        self,
        vectoriser_uri: str,
        target_feature: Union[Features, str] = Features.TFIDF_FULL_TEXT,
        source_feature: Union[Features, str] = Features.ARTICLE_TEXT,
    ):
        super().__init__()
        self.source_feature = source_feature
        self.target_feature = target_feature
        self._load_vectoriser(vectoriser_uri)

    def _load_vectoriser(self, uri: str) -> None:
        print(f"Starting loading vectoriser from {uri}")
        self.vectoriser = pickle.load(open(uri, "rb"))
        if not callable(getattr(self.vectoriser, "transform", None)):
            raise ValueError("Given vectoriser must have a callable attribute 'transform'")
        print("Done loading the vectoriser")

    def visit_article(self, article: Article) -> None:
        """
        Visit the article and apply the vectorisation.

        Parameters
        ----------
        article:
            The article to be visited.

        """
        clean_text = replace_entity_name(
            article.extracted_entities[self.source_feature],
            article.entity_name,
            "default-experiment",
        )
        article.extracted_entities[self.target_feature] = self.vectoriser.transform([clean_text])


class FullSanctionTextVectoriser(SanctionVisitor):

    """
    Vectorises all available sanction text.

    The vectorised form can be used by downstream ML components.

    Attributes
    ----------
    vectoriser: object
        A vectoriser given must implement transform method.
        The methods must take a list of articles and output a vector form of a given article.

    """

    def __init__(
        self,
        vectoriser_uri: str,
        target_feature: Union[Features, str] = SanctionFeatures.TFIDF,
        cache: str = "",
    ):
        super().__init__()
        self.target_feature = target_feature
        self.cache = cache
        self._load_vectoriser(vectoriser_uri)

    def _load_vectoriser(self, uri: str) -> None:
        uri = ensure_s3_resource_exists(uri=uri, target_folder=self.cache)

        print(f"Starting loading vectoriser from {uri}")
        self.vectoriser = pickle.load(open(uri, "rb"))
        if not callable(getattr(self.vectoriser, "transform", None)):
            raise ValueError("Given vectoriser must have a callable attribute 'transform'")
        print("Done loading the vectoriser")

    @staticmethod
    def _get_text_from_raw_entity(raw_entity: Dict):
        text_tokens = []
        if raw_entity["data"]["display_fields"] is not None:
            for item in raw_entity["data"]["display_fields"]:
                text_tokens.append(f'{item["value"]}.')
        if "occupations" in raw_entity["data"]:
            if raw_entity["data"]["occupations"] is not None:
                for item in raw_entity["data"]["occupations"]:
                    if "occupation" in item:
                        text_tokens.append(f'{item["occupation"]}.')
        return " ".join(text_tokens)

    def visit_sanction(self, sanction: Sanction) -> None:
        """
        Visit the article and apply the vectorisation.

        Parameters
        ----------
        sanction:
            The sanction to be visited.

        """
        clean_text = self._get_text_from_raw_entity(sanction.raw_entity)
        sanction.extracted_entities[self.target_feature] = self.vectoriser.transform([clean_text])


class TFIDFFullTextVisitorS3(FullArticleTextVectoriser):

    """

    Same as the parent class, but can download vectoriser model from S3.

    Attributes
    ----------
    vectoriser_uri: str
        S3 locator of the model file
    cache: str
        Folder to store and cache downloaded models.

    """

    def __init__(
        self,
        vectoriser_uri: str,
        target_feature: Union[Features, str] = Features.TFIDF_FULL_TEXT,
        source_feature: Union[Features, str] = Features.ARTICLE_TEXT,
        cache: str = "",
    ):
        self.cache = cache
        super().__init__(vectoriser_uri, target_feature, source_feature)

    def _load_vectoriser(self, uri: str) -> None:
        model_fn = ensure_s3_resource_exists(uri=uri, target_folder=self.cache)

        super()._load_vectoriser(model_fn)


class FullArticleFeaturesTextExtractor(ArticleVisitor):

    """
    Vectorises the full article text.

    The vectorised form can be used by downstream ML components.

    """

    def visit_article(self, article: Article) -> None:
        """
        Visit the article and apply the vectorisation.

        Parameters
        ----------
        article:
            The article to be visited.

        """
        article.extracted_entities[Features.FULL_TEXT_FEATURES] = get_features_from_article(article)


class FullArticleFeaturesTextVectoriser(FullArticleTextVectoriser):

    """
    Vectorises the full article text features.

    The vectorised form can be used by downstream ML components.

    """

    def visit_article(self, article: Article) -> None:
        """
        Visit the article and apply the vectorisation.

        Parameters
        ----------
        article:
            The article to be visited.

        """
        article.extracted_entities[Features.TFIDF_FULL_TEXT_FEATURES] = self.vectoriser.transform(
            [article.extracted_entities[Features.FULL_TEXT_FEATURES]]
        )


class BertVectoriser:

    """
    Vectorises the full article text using Bert.

    The vectorised form can be used by downstream ML components.

    """

    def __init__(
        self,
        target_feature: Union[Features, str] = Features.TFIDF_FULL_TEXT,
        source_feature: Union[Features, str] = Features.ARTICLE_SENTENCES,
    ):
        self.model = SentenceTransformer("bert-base-nli-max-tokens")
        self.target_feature = target_feature
        self.source_feature = source_feature

    def visit_article(self, article: Article) -> None:
        """
        Get a vectorised representation of the text using bert.

        Obtain bert vectorised tokens and get a mean of all the sentences embeddings.

        Parameters
        ----------
        article:
            The article to be vectorised.

        """
        sentences = article.extracted_entities[self.source_feature]
        if sentences:
            sentence_embeddings = self.model.encode(sentences)
            article_text_encoding = np.mean(sentence_embeddings, axis=0)
            article.extracted_entities[self.target_feature] = csr_matrix(article_text_encoding)
        else:
            raise ValueError("No sentences to process")


class JsonSummarizer(SanctionVisitor):

    """Summarizes a given json into a string."""

    def __init__(self):
        super().__init__()
        self.display_fields_blocks = {
            "Amended On",
            "Designation Act",
            "Function",
            "Other Information",
            "Program",
            "Related Url",
            "Designation Date",
            "Issuing Authority",
            "Listing Id",
            "Title",
            "Listing Origin",
            "Reason",
            "Sanction Type",
            "Un Listing Id",
            "Additional",
            "Other Info",
            "Citizenship",
            "List Id",
            "Address",
            "Identification Number",
            "List Name",
            "Enforcement Agency",
            "Legal Basis",
            "Position",
            "Regime",
            "Removal Date",
            "Remark",
            "Role",
            "Registration Number",
            "Zip Code",
            "Height",
            "NI Number",
            "Ofsi Listing Id",
            "Sanctions Type",
            "Unique Id",
            "Designating Authority",
            "Declaration",
            "Description",
            "Regulation",
            "Comments",
            "Program Entry",
            "UN List Type",
            "OFAC ID",
            "Programs",
            "Additional Sanctions Information",
            "Designation",
            "Justification",
            "Known Addresses",
            "Basis",
            "Committees",
            "ROSFIN Description",
            "Ministerial Decision Date",
            "Source",
            "Additional Information",
            "Listing Information",
            "National ID No",
            "Special Economic Measure Act",
            "National Id",
            "Addresses",
            "Date Listed",
            "Language",
            "Contact Details",
            "Digital Currency Address",
            "Document ID",
            "Cedula No",
            "National Register Number",
            "Listing Category",
            "SSN",
            "Birthplace",
            "Programme",
        }

    def visit_sanction(self, sanction: Sanction) -> None:
        """Summarize a given json into text."""
        text_tokens = []
        df = (
            sanction.raw_entity["data"]["display_fields"]
            if sanction.raw_entity["data"]["display_fields"]
            else []
        )
        for item in df:
            if item["title"] not in self.display_fields_blocks:
                continue
            text_tokens.append(item["value"])

        sanction.extracted_entities[SanctionFeatures.FULL_TEXT] = ".".join(text_tokens)
