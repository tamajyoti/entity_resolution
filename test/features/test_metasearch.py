from am_combiner.features.article import Article, Features
from am_combiner.features.metadata_search import MetaKeyVisitor


def test_metadata_search():
    article = Article(
        entity_name="Charlie",
        article_text="check charlie for adverse media",
        url="test@charlie.com",
        meta={"listing_subtype": "adverse-media-v2-violence-aml-cft"},
    )

    article.accept_visitor(MetaKeyVisitor("listing_subtype", Features.AM_CATEGORY))

    assert article.meta == {"listing_subtype": "adverse-media-v2-violence-aml-cft"}
    assert Features.AM_CATEGORY in article.extracted_entities
    assert article.extracted_entities[Features.AM_CATEGORY] == {"adverse-media-v2-violence-aml-cft"}
