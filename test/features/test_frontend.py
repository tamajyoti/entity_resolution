import pytest

from am_combiner.features.article import Features
from am_combiner.features.sanction import SanctionFeatures
from am_combiner.features.common import SpacyArticleVisitor, SanctionPrimariesExtractor
from am_combiner.features.frontend import (
    ArticleFeatureExtractorFrontend,
    _process_article,
    _process_sanction,
)


class TestFeatureExtractionFrontend:
    def test_frontend_can_be_initialised_empty(self):
        ArticleFeatureExtractorFrontend(visitors_cache={}, visitors=[], thread_count=4)

    def test_frontend_rises_exception_if_visitor_not_in_cache(self):
        with pytest.raises(ValueError):
            ArticleFeatureExtractorFrontend(
                visitors_cache={
                    "A": SpacyArticleVisitor(),
                },
                visitors=["C"],
                thread_count=4,
            )

    def test_articles_get_visited(self, test_dataframe):
        visited_article = _process_article(
            (
                ("John Smith is a nice guy", "http://nice.com", "John Smith", {}),
                [SpacyArticleVisitor()],
            )
        )
        assert Features.PERSON in visited_article.extracted_entities

    def test_sanctions_get_visited(self):
        visitor = SanctionPrimariesExtractor()
        visited_sanction = _process_sanction(
            (
                ("_id1", {"data": {"names": []}}, ""),
                [visitor],
            )
        )
        assert SanctionFeatures.PRIMARY in visited_sanction.extracted_entities
