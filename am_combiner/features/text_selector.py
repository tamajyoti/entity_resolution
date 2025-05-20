from typing import List
from am_combiner.features.article import Features, Article
from am_combiner.features.common import ArticleVisitor


def previous_and_next(
    sentence_list: List[str], entity_name: str, pre: int = 1, post: int = 1
) -> str:
    """
    Obtain N sentences pre and post the entity name in article.

    :param sentence_list:
        List of sentences.
    :param entity_name:
        Name of the entity.
    :param pre:
        No of sentences to be selected previous
    :param post:
        No of sentences to be selected post
    :return:
        Joined set of sentences that are to be considered.
    """
    output = []
    unique = set()
    for i, s in enumerate(sentence_list):
        if entity_name not in s:
            continue

        start = max(0, i - pre)
        finish = min(len(sentence_list), i + post + 1)

        new_sents = sentence_list[start:finish]
        for ns in new_sents:
            if ns in unique:
                continue
            output.append(ns)
        unique.update(new_sents)

    return " ".join(output)


class ArticleSelectedTextVisitor(ArticleVisitor):

    """
    A concrete implementation of the ArticleVisitor class.

    Selects text from the article text.
    """

    def __init__(
        self,
        source_feature: Features = Features.ARTICLE_SENTENCES,
        target_feature: Features = Features.ARTICLE_TEXT_SELECTED,
        preceding_sentences: int = 1,
        post_sentences: int = 1,
    ):
        super().__init__()
        self.source_feature = source_feature
        self.target_feature = target_feature
        self.prev_sent = preceding_sentences
        self.post_sent = post_sentences

    def visit_article(self, article: Article) -> None:
        """
        Call spaCu term extractor pipeline which does keyword extraction.

        We then only save top self.n keywords for feature analysis.

        Parameters
        ----------
        article:
            Article object to be modified.

        """
        sentences = article.extracted_entities[self.source_feature]
        selected_text = previous_and_next(
            sentences, article.entity_name, self.prev_sent, self.post_sent
        )

        article.extracted_entities[self.target_feature] = selected_text
