from am_combiner.features.article import Article, Features
from am_combiner.features.common import SpacyArticleVisitor


class TestCoreferenceResolutionVisitor:
    def test_coref_resolution_init(self):
        visitor = SpacyArticleVisitor(do_coref_resolution=True)
        article = Article(
            entity_name="Sister", article_text="My sister has a dog. She loves him", url=None
        )
        article.accept_visitor(visitor)
        assert Features.COREFERENCE_RESOLVED_TEXT in article.extracted_entities
        assert Features.COREFERENCE_RESOLVED_CLUSTERS in article.extracted_entities
        assert (
            article.extracted_entities[Features.COREFERENCE_RESOLVED_TEXT]
            == "My sister has a dog. My sister loves a dog"
        )
