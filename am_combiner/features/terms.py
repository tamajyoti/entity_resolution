import pandas as pd
from pyate import combo_basic

from am_combiner.features.article import Features, Article
from am_combiner.features.common import ArticleVisitor


class ArticleTermVisitor(ArticleVisitor):

    """
    A concrete implementation of the ArticleVisitor class.

    Runs a spaCy term extraction pipeline on the article text.
    Selects top terms for future analysis.

    Attributes
    ----------
    n: int
        Defines how many top terms to take from the extracted list.

    """

    def __init__(self, n: int = 10):
        """
        Initialize an article term visitor.

        Parameters
        ----------
        n:
            Will take top n keywords extracted.

        """
        super().__init__()
        self.n = n

    def visit_article(self, article: Article) -> None:
        """
        Call spaCu term extractor pipeline which does keyword extraction.

        We then only save top self.n keywords for feature analysis.

        Parameters
        ----------
        article:
            Article object to be modified.

        """
        all_terms = combo_basic(article.extracted_entities[Features.ARTICLE_TEXT])
        best_terms = all_terms.sort_values(ascending=False).head(self.n).index.to_list()
        best_terms = [t.lower() for t in best_terms]

        article.extracted_entities[Features.TERM] = best_terms


class ArticleKeywordVisitor(ArticleVisitor):

    """
    Looks up keywords from a pre-defined list of terms in an article text.

    Attributes
    ----------
    keywords_filename: str
        A path to the list of keywords. The file is assumed to be 1-column headless csv.
    feature_name: str
        The name of the key of the dictionary containing the extracted entities.

    """

    def __init__(self, keywords_filename: str, feature_name: Features):
        """
        Initialize an article keyword visitor.

        Parameters
        ----------
        keywords_filename:
            Path to the file containing a single headless column with a list of keywords
            to be detected.
        feature_name:
            The name of the key in the article extracted entities dictionary containing
            the extracted keywords.

        """
        super().__init__()
        self.keywords = set(
            pd.read_csv(keywords_filename, header=None)[0].str.strip().str.lower().to_list()
        )
        self.feature_name = feature_name

    def visit_article(self, article: Article) -> None:
        """
        Detect keyword from self.keywords in the article text.

        Parameters
        ----------
        article:
            An article object to be modified.

        """
        lowered_article = article.extracted_entities[Features.ARTICLE_TEXT].lower()
        present_keywords = [ele for ele in self.keywords if (ele in lowered_article)]
        article.extracted_entities[self.feature_name] = present_keywords
