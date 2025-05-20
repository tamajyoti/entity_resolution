from am_combiner.features.article import Features
from am_combiner.features.common import SpacyArticleVisitor, EntityNameRemoverVisitor


def test_name_remover_removes_entity_name(article):
    entity = "John Smith"
    block = "John Smith was a news anchor and a psychotherapist"
    test_article = article(entity, block)
    test_article.accept_visitor(SpacyArticleVisitor())
    test_article.accept_visitor(EntityNameRemoverVisitor())
    assert (
        test_article.extracted_entities[Features.ARTICLE_TEXT]
        == "  was a news anchor and a psychotherapist"
    )


def test_name_remover_removes_two_names(article):
    entity = "John Smith"
    block = (
        "John Smith was a news anchor and a psychotherapist, just like his father, Robert Peterson"
    )
    test_article = article(entity, block)
    test_article.accept_visitor(SpacyArticleVisitor())
    test_article.accept_visitor(EntityNameRemoverVisitor())
    assert (
        test_article.extracted_entities[Features.ARTICLE_TEXT]
        == "  was a news anchor and a psychotherapist, just like his father,  "
    )
