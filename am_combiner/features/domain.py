from urllib.parse import urlparse
from am_combiner.features.common import ArticleVisitor
from am_combiner.features.article import Article, Features


class UrlDomainVisitor(ArticleVisitor):

    """Retrieve domain from url."""

    def __init__(self):
        """Initialize an article url domain visitor."""
        super().__init__()

    def visit_article(self, article: Article) -> None:
        """
        Extract domain information from url.

        Parameters
        ----------
        article:
            An article object to be modified.

        """
        domain = urlparse(article.url).netloc
        if domain:
            article.extracted_entities[Features.DOMAIN] = set([domain])
