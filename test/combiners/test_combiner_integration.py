import numpy as np
from scipy import sparse
import yaml
import pytest

from am_combiner.utils.parametrization import get_cache_from_yaml, features_str_to_enum
from am_combiner.features.article import Features, Article
from am_combiner.combiners.mapping import COMBINER_CLASS_MAPPING
from am_combiner.combiners.common import CLUSTER_ID_FIELD


def gen_sparse_matrix(shape):
    return sparse.csr_matrix(np.random.random(shape).clip(0.8) - 0.8)


def get_test_articles(n):

    # Initiate Articles
    articles = []
    for i in range(n):
        articles.append(Article("A", "", f"url{i}"))

    np.random.seed(0)
    for i, article in enumerate(articles):

        # Add TFIDF features:
        tfidf_value = gen_sparse_matrix(8000)
        for f in [
            Features.TFIDF_FULL_TEXT,
            Features.TFIDF_COREFERENCE_RESOLVED_TEXT,
            Features.TFIDF_SELECTED_TEXT,
            Features.TFIDF_FULL_TEXT_8000,
        ]:
            article.extracted_entities[f] = tfidf_value

        article.extracted_entities[Features.TFIDF_FULL_TEXT_12000] = gen_sparse_matrix(12000)

        # Add BERT feature:
        article.extracted_entities[Features.BERT_FEATURES] = gen_sparse_matrix(768)

        # Add edge features:
        for f in [Features.ORG_CLEAN, Features.PERSON_CLEAN, Features.GPE_CLEAN, Features.LOC]:
            article.extracted_entities[f] = np.random.choice(range(10), 3)

    return articles


@pytest.mark.integtest
def test_all_combiners_on_fake_data():

    articles = get_test_articles(5)

    combiner_cache = get_cache_from_yaml(
        "combiners_config.yaml",
        section_name="combiners",
        class_mapping=COMBINER_CLASS_MAPPING,
        restrict_classes=set(),
        attrs_callbacks={
            "source_feature": features_str_to_enum,
            "node_features": features_str_to_enum,
            "use_features": lambda fs: [features_str_to_enum(f) for f in fs],
        },
    )
    with open("combiners_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Check that all combiners have been loaded:
    assert len(combiner_cache) == len(config["combiners"])

    for combiner_name, combiner_object in combiner_cache.items():
        if combiner_name != "AnnotationsCombiner":
            combiner_df = combiner_object.combine_entities(articles)
            assert combiner_df.shape[0] == 5
            assert combiner_df[CLUSTER_ID_FIELD].min() == 0
            assert combiner_df[CLUSTER_ID_FIELD].max() <= 4
