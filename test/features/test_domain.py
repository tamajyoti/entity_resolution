import pytest

from am_combiner.features.domain import UrlDomainVisitor
from am_combiner.features.article import Features, Article


@pytest.mark.parametrize(
    ["article", "domain"],
    [
        (Article("", ""), set()),
        (Article("", "", "https://www.bbc.co.uk/news/technology"), set(["www.bbc.co.uk"])),
        (Article("", "", "https://www.delfi.lt/miestai/vilnius/"), set(["www.delfi.lt"])),
        (Article("", "", "This is not a url"), set()),
        (Article("", "", ""), set()),
    ],
)
def test_url_domain(article, domain):
    visitor = UrlDomainVisitor()
    article.accept_visitor(visitor)
    assert article.extracted_entities[Features.DOMAIN] == domain
