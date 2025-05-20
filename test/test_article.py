from operator import itemgetter

from am_combiner.features.article import Features
from am_combiner.features.common import SpacyArticleVisitor, FieldCleaningVisitor
from am_combiner.features.terms import ArticleTermVisitor, ArticleKeywordVisitor


def test_dump_article_terms(article):
    entity = "King George III"
    block = (
        "The Geographical collection of King George III includes the King’s Topographical "
        "collection of maps, views and atlases, and the King’s Maritime collection of "
        "sea charts."
    )
    test_article = article(entity, block)
    test_article.accept_visitor(ArticleTermVisitor())
    assert sorted(test_article.dump_tuples(), key=itemgetter(0)) == [
        ("geographical collection", Features.TERM),
        ("maritime collection", Features.TERM),
        ("sea charts", Features.TERM),
        ("topographical collection", Features.TERM),
    ]


def test_dump_field_after_field_cleaning(article):
    visitor = FieldCleaningVisitor(feature_name=Features.PERSON, target_feature=Features.PERSON)
    entity = "John Smith"
    block = "John Smith is a nice guy and Mary Jones is a cute gal."
    test_article = article(entity, block)
    test_article.accept_visitor(SpacyArticleVisitor())
    test_article.accept_visitor(visitor)
    assert test_article.dump_tuples() == [
        ("Mary Jones", Features.PERSON),
        ("John Smith is a nice guy and Mary Jones is a cute gal.", Features.ARTICLE_SENTENCES),
    ]


def test_dump_article_after_spacy(article):
    entity = "John Smith"
    block = "John Smith is a nice guy"
    test_article = article(entity, block)
    test_article.accept_visitor(SpacyArticleVisitor())
    assert test_article.dump_tuples() == [
        ("John Smith", Features.PERSON),
        ("John Smith is a nice guy", Features.ARTICLE_SENTENCES),
    ]


def test_dump_article_keywords(article):
    visitor = ArticleKeywordVisitor(
        keywords_filename="am_combiner/data/keywords.csv", feature_name=Features.TERM_KEYWORD
    )
    entity = "John Smith"
    block = "John Smith kept his victims in inhumane conditions"
    test_article = article(entity, block)
    test_article.accept_visitor(visitor)
    assert set(test_article.dump_tuples()) == {
        ("victim", Features.TERM_KEYWORD),
        ("inhumane", Features.TERM_KEYWORD),
    }


def test_dump_article_profession_nltk_keywords(article):
    visitor = ArticleKeywordVisitor(
        keywords_filename="am_combiner/data/professions_nltk.csv",
        feature_name=Features.PROFESSION_NLTK_KEYWORD,
    )
    entity = "John Smith"
    block = "John Smith was a news anchor and a psychotherapist"
    test_article = article(entity, block)
    test_article.accept_visitor(visitor)
    assert set(test_article.dump_tuples()) == {
        ("therapist", Features.PROFESSION_NLTK_KEYWORD),
        ("psychotherapist", Features.PROFESSION_NLTK_KEYWORD),
    }


def test_dump_article_profession_keywords(article):
    visitor = ArticleKeywordVisitor(
        keywords_filename="am_combiner/data/occupation_list_global.csv",
        feature_name=Features.PROFESSION_KEYWORD_KEYWORD,
    )
    entity = "John Smith"
    block = "John Smith was a travelling showman, accounts staff and an underwriter"
    test_article = article(entity, block)
    test_article.accept_visitor(visitor)
    assert set(test_article.dump_tuples()) == {
        ("travelling showman", Features.PROFESSION_KEYWORD_KEYWORD),
        ("accounts staff", Features.PROFESSION_KEYWORD_KEYWORD),
        ("showman", Features.PROFESSION_KEYWORD_KEYWORD),
        ("underwriter", Features.PROFESSION_KEYWORD_KEYWORD),
        ("writer", Features.PROFESSION_KEYWORD_KEYWORD),
        ("staff", Features.PROFESSION_KEYWORD_KEYWORD),
    }
