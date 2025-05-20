from itertools import chain
from typing import Union

import gensim
from gensim import corpora
from scipy.sparse import csr_matrix

from am_combiner.features.article import Features, Article
from am_combiner.features.common import ArticleVisitor
from am_combiner.utils.topic_model_helpers import (
    remove_stopwords,
    get_article_words,
    make_bigrams,
    lemmatization,
    missing_topics,
)
from am_combiner.utils.storage import ensure_s3_resource_exists


class TopicVisitor(ArticleVisitor):

    """
    Checks the topic distribution of a documents and return the topic ids.

    Attributes
    ----------
    lda_model_path: str
        A path to the lda_mdodel
    dictionary_path: str
        A path to the corpus dictionary

    """

    def __init__(
        self,
        lda_model_path: str,
        dictionary_path: str,
        bigram_path: str,
        state_path: str,
        beta_path: str,
        cache: str,
        feature_topic: Union[Features, str],
        feature_distribution: Union[Features, str],
    ):
        """
        Initialize an article keyword visitor.

        Parameters
        ----------
        lda_model_path: str
            A path to the lda_mdodel
        dictionary_path: str
            A path to the corpus dictionary
        state_path: str
            Path to model state
        beta_path: str
            Path to model beta
        bigram_path: str
            Path to bigram module
        cache: str
            Path to cache folder
        feature_distribution:
            distribution of feature
        feature_topic:
            distribution of topic

        """
        super().__init__()
        # Download LDA files from S3
        self.lda_model_path = ensure_s3_resource_exists(lda_model_path, target_folder=cache)
        self.dictionary_path = ensure_s3_resource_exists(dictionary_path, target_folder=cache)
        # Download tertiary files
        ensure_s3_resource_exists(state_path, target_folder=cache)
        ensure_s3_resource_exists(beta_path, target_folder=cache)
        ensure_s3_resource_exists(bigram_path, target_folder=cache)

        self.feature_topic = feature_topic
        self.feature_distribution = feature_distribution

        self.lda_model = gensim.models.ldamodel.LdaModel.load(str(self.lda_model_path))
        self.word_dic = corpora.Dictionary.load(f"{self.lda_model_path}.id2word")
        self.cache = cache

    def visit_article(self, article: Article) -> None:
        """
        Detect keyword from self.keywords in the article text.

        Parameters
        ----------
        article:
            An article object to be modified.

        """
        # Get article text sentences and preprocess
        sent_lists = [article.extracted_entities[Features.ARTICLE_SENTENCES]]
        data_words_nostops = remove_stopwords(get_article_words(sent_lists))
        words_bigrams = make_bigrams(data_words_nostops, self.cache)
        words_lemmatized = lemmatization(
            words_bigrams, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]
        )

        # Tranform the text and get the topic ouput
        texts = words_lemmatized
        other_corpus = [self.word_dic.doc2bow(text) for text in texts]

        topic_output = self.lda_model[other_corpus[0]][0]

        # get topic id list

        topic_ids = set([val[0] for val in topic_output])
        missed_topics = missing_topics(20, topic_output)

        topic_vector = sorted(list(chain(topic_output, missed_topics)))

        article.extracted_entities[self.feature_topic] = topic_ids

        article.extracted_entities[self.feature_distribution] = csr_matrix(
            [val[1] for val in topic_vector]
        )
