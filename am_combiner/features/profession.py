# AUTOGENERATED! DO NOT EDIT! File to edit: 03_Feature_Profession.ipynb (unless otherwise specified).

__all__ = ['nlp', 'check_words', 'ProfessionFeatureExtractor', 'ProfessionVisitor']

# Cell
import spacy
import pandas as pd
from typing import List
from .article import Article, Features
from .common import ArticleVisitor, TextCleaningVisitor

nlp = spacy.load("en_core_web_sm")


# Cell
def check_words(sentence: List[str], words: List[str]) -> List[str]:
    """
    The check function checks a specific list of words in a sentence

    Parameters:
    -----------
    sentence: list,
        The list of sentences to be searched
    words: list,
        The list of words which are to be searched

    Returns:
    --------
    sentences: list,
        The list of sentences which have the words that were searched for

    """
    res = [any([k in s for k in words]) for s in sentence]
    return [sentence[i] for i in range(0, len(res)) if res[i]]


# Cell
class ProfessionFeatureExtractor:
    """
    The class lists all the patterns used to identify the various ways to obtain featureX

    We initialize the class by a spacy transformed text and the entity name and a list of occupations obtained
    from the occupation list stored as a csv

    ...

    Attributes
    ----------
    text: str
        The string representing article text. Just a normal raw text, nothing else.
    entity_name: str
        The entity whose article texts we are transforming to obtaining the feature
    occupation_list: list
        The global list of occupations obtained and modified from an external CSV file

    Methods
    -------
    pattern_subj(self)
        Accepts the initialized class for the text which needs to checked. Check from the NSUBJ occurrence and
        then returns the noun chunks containing the NSUBJ

    pattern_appos(self)
        Accepts the initialized class for the text which needs to checked. Check from the APPOS occurrence and
        then returns the noun chunks APPOS

    """

    def __init__(self, text: str, entity_name: str, occupation_list: List[str]) -> None:
        """
        the initialization function to initialize the class

        Parameters:
        ------------
        text: str,
            The text which needs to be spacy transformed
        entity_name: str,
            The entity_name that for which the feature is to identified
        occupation_list: list,
            The list of occupations created and served as a CSV while initializing the class

        Returns:
        ---------
        feature_class:
            Initialized class
        """
        self.text = nlp(text)
        self.words = [y for y in entity_name.split()]
        self.occupation = occupation_list

    def pattern_subj(self) -> List[str]:
        """
        The pattern check iterates on the spacy transformed text's noun chunks for a NSUBJ spacy dependency
        which contains the mention of the entity name and then check whether the subject dependency
        contains any mention of occupation from the occupation list

        Parameters:
        ------------
        self:
            The initialized class parameters

        Returns:
        ----------
        chunks: list,
            List of chunks which pass the conditional check
        """
        chunks = []
        for chunk in self.text.noun_chunks:
            if (chunk.root.dep_ == "nsubj") & (any(word in chunk.text for word in self.words) == True) \
                    & (any(word.lower() in chunk.text.lower() for word in self.occupation) == True):
                chunks.append(chunk.text)
        return chunks

    def pattern_appos(self) -> List[str]:
        """
        The pattern check iterates on the spacy transformed text's noun chunks for a APPOS spacy dependency
        then check whether the subject dependency contains any mention of occupation from the occupation list

        Parameters:
        ------------
        self:
            The initialized class parameters

        Returns:
        ----------
        chunks: list,
            List of chunks which pass the conditional check
        """
        chunks = []
        for chunk in self.text.noun_chunks:
            if (chunk.root.dep_ == "appos") \
                    & (any(word.lower() in chunk.text.lower() for word in self.occupation) == True):
                chunks.append(chunk.text)
        return chunks


# Cell
class ProfessionVisitor(ArticleVisitor):
    """
    This class is a concrete implementation of a visitor pattern.
    This implementation makes sure that any tokens in entity name are not part of extracted PERSON/ORG tuples.

    Attributes
    ----------
        field_name: class
            the name of the field is automatically obtained from feature class attribute for profession
            visitor. It is obtained during the run of the main file
        occupation_csv_path: str
            The exact file path containing the occupation names csv

    """

    def __init__(self, field_name: Features, occupation_csv_path: str) -> None:
        """
        the initialization function to initialize the class

        Parameters:
        ------------
        field_name: class
            the name of the field is automatically obtained from feature class attribute for profession
            visitor. It is obtained during the run of the main file
        occupation_csv_path: str,
            The exact file path containing the occupation names csv

        Returns:
        ---------
        feature_class:
            Initialized class
        """
        super().__init__()
        self.field_name = field_name
        self.occupation = list(pd.read_csv(
            occupation_csv_path, header=None)[0])

    def visit_article(self, article: Article) -> None:
        """
        The visit article get the initialized article and then created the feature named by the field_name
        which is passed in the parameter.

        It obtains the article text and then then calls the various spacy transformation and pattern
        matching functions to obtain the profession feature for an entity

        Parameters
        ----------
        article: Article
            The initialized article class from which the feature needs to be extracted

        Returns
        -------
        None
        """
        # converting the article text into sentences and then selecting only those sentences
        # containing the entity name using the check_words function

        article.accept_visitor(TextCleaningVisitor())
        doc = nlp(list(article.extracted_entities.items())[0][1])
        sentences = [sent.string.strip() for sent in doc.sents]
        words = [y for y in article.entity_name.split()]
        imp_sentences = check_words(sentences, words)

        all_prof = []
        for sent in imp_sentences:
            text_noun_chunks = ProfessionFeatureExtractor(
                sent, article.entity_name, self.occupation)
            all_prof.append(text_noun_chunks.pattern_subj())
            all_prof.append(text_noun_chunks.pattern_appos())
        feature_prof = [item for sublist in all_prof for item in sublist]
        article.extracted_entities[f"{self.field_name}"] = feature_prof
