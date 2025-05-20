from am_combiner.features.article import Features, Article
from am_combiner.features.common import ArticleVisitor


class MetaKeyVisitor(ArticleVisitor):

    """
    Looks up keywords from a pre-defined list of terms in an article text.

    Attributes
    ----------
    feature_name: str
        The name of the key of the dictionary containing the extracted entities.

    """

    def __init__(self, feature_key: str, feature_name: Features):
        """
        Initialize an article keyword visitor.

        Parameters
        ----------
        feature_key:
            The exact metadata key which needs to be extracted
        feature_name:
            The name of the key in the article extracted entities dictionary containing
            the extracted keywords.

        """
        super().__init__()
        self.key = feature_key
        self.feature_name = feature_name

    def visit_article(self, article: Article) -> None:
        """
        Detect keyword from self.keywords in the article text.

        Parameters
        ----------
        article:
            An article object to be modified.

        """
        if self.key in article.meta:
            article.extracted_entities[self.feature_name] = {article.meta[self.key]}
        else:
            article.extracted_entities[self.feature_name] = {}
