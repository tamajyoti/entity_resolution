import os
from typing import DefaultDict, List
import spacy
import nltk
from gensim.models.phrases import Phraser
from nltk.corpus import stopwords
import gensim
from gensim.utils import simple_preprocess

from am_combiner.features.article import Features

nltk.download("stopwords")
PATH = "am_combiner/data/topic_model/"
NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def sent_to_words(sentences: List):
    """Generate words from sentences."""
    for sentence in sentences:
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)


def get_article_list(
    article_list: DefaultDict, min_sent_len: int = 20, max_sent_len: int = 70, skip_flag=True
):
    """Get list of article and their sentences."""
    list_articles = []
    for key in article_list.keys():
        for article in article_list[key]:
            sent_len = len(article.extracted_entities[Features.ARTICLE_SENTENCES])
            if skip_flag:
                if min_sent_len <= sent_len <= max_sent_len:
                    list_articles.append(article.extracted_entities[Features.ARTICLE_SENTENCES])
            else:
                list_articles.append(article.extracted_entities[Features.ARTICLE_SENTENCES])
    return list_articles


def get_article_words(article_list: List):
    """Get words tokenized from article."""
    all_article_words = []
    for article in article_list:
        data_words = list(sent_to_words(article))
        all_words = [word for words in data_words for word in words]
        all_article_words.append(all_words)

    return all_article_words


def get_ngram_model(all_article_words: List[List]):
    """Get bigrams from article."""
    bigram = gensim.models.Phrases(all_article_words, min_count=5, threshold=100)
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)

    bigram_path = os.path.join(PATH, "bigram_phraser.pickle")
    bigram_mod.save(bigram_path)

    return None


# Define functions for stopwords, bigrams, trigrams and lemmatization
# Define global nltk and spacy implementations


def remove_stopwords(texts):
    """Remove stopwords."""
    stop_words = stopwords.words("english")
    stop_words.extend(["from", "subject", "re", "edu", "use"])
    return [
        [word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts
    ]


def make_bigrams(texts: List[List[str]], path=PATH):
    """Make bigrams."""
    bigram_path = os.path.join(path, "bigram_phraser.pickle")
    bigram_mod = Phraser.load(bigram_path)

    return [bigram_mod[doc] for doc in texts]


def lemmatization(texts: List[List[str]], allowed_postags=["NOUN", "ADJ", "VERB", "ADV"], nlp=NLP):
    """https://spacy.io/api/annotation."""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def missing_topics(total_topics, obtained_topics):
    """Pad the data with missing topics."""
    total_topics = set(list(range(total_topics)))
    obtained_topics = set([val[0] for val in obtained_topics])
    missed_topics = total_topics - obtained_topics
    missing_topic_vector = []
    for topic in missed_topics:
        data_tuple = (topic, 0)
        missing_topic_vector.append(data_tuple)

    return missing_topic_vector
