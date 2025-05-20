import os
from typing import DefaultDict

import gensim
from gensim import corpora

from am_combiner.utils.topic_model_helpers import (
    remove_stopwords,
    get_article_words,
    get_article_list,
    make_bigrams,
    lemmatization,
)

PATH = "am_combiner/data/topic_model/"


def topic_model_train(entity_articles: DefaultDict, topics: int = 20):
    """Build the topic model."""
    data_words_nostops = remove_stopwords(get_article_words(get_article_list(entity_articles)))
    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)
    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(
        data_words_bigrams, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]
    )

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    lda_model = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=topics,
        random_state=100,
        update_every=1,
        chunksize=100,
        passes=10,
        alpha="auto",
        per_word_topics=True,
    )

    # create the paths to store the model,dictionary and phrases
    model_path = os.path.join(PATH, f"lda_model_{topics}")

    lda_model.save(model_path)
