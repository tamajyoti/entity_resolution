from am_combiner.features.article import Article, Features
from am_combiner.features.common import SpacyArticleVisitor
from am_combiner.features.text_selector import ArticleSelectedTextVisitor


class TestSelectedTextVisitor:
    def test_selected_text_init(self):
        full_text = """Charlie in this sentence.This sentence of him.This sentence not of him."""
        spacy_visitor = SpacyArticleVisitor()
        selected_visitor = ArticleSelectedTextVisitor()
        article = Article(
            entity_name="Charlie",
            article_text=full_text,
            url=None,
        )
        article.accept_visitor(spacy_visitor)
        article.accept_visitor(selected_visitor)

        assert Features.ARTICLE_SENTENCES in article.extracted_entities
        assert article.extracted_entities[Features.ARTICLE_SENTENCES] == [
            "Charlie in this sentence.",
            "This sentence of him.",
            "This sentence not of him.",
        ]

        assert Features.ARTICLE_TEXT_SELECTED in article.extracted_entities
        assert (
            article.extracted_entities[Features.ARTICLE_TEXT_SELECTED]
            == """Charlie in this sentence. This sentence of him."""
        )
