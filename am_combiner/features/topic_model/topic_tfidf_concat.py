from typing import Union

from scipy.sparse import hstack
from am_combiner.features.article import Features, Article
from am_combiner.features.common import ArticleVisitor


class TopicTfidfConcatVisitor(ArticleVisitor):

    """
    Concat topic ids and tfids in a single vector.

    Attributes
    ----------
    feature_topic: str
        A path to the lda_mdodel
    feature_tfidf: str
        A path to the corpus dictionary

    """

    def __init__(
        self,
        feature_topic: Union[Features, str],
        feature_tfidf: Union[Features, str],
        target_feature: Union[Features, str],
    ):
        """
        Initialize an article keyword visitor.

        Parameters
        ----------
        feature_topic:
            name of topic feature
        feature_tfidf:
            name of tfidf feature
        target_feature:
            final concatenated feature of topic and tfidf

        """
        super().__init__()

        self.feature_topic = feature_topic
        self.feature_tfidf = feature_tfidf
        self.target_feature = target_feature

    def visit_article(self, article: Article) -> None:
        """
        Detect keyword from self.keywords in the article text.

        Parameters
        ----------
        article:
            An article object to be modified.

        """
        article.extracted_entities[self.target_feature] = hstack(
            (
                article.extracted_entities[self.feature_tfidf],
                article.extracted_entities[self.feature_topic],
            )
        ).tocsr()
