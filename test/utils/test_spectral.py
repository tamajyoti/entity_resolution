import numpy as np
import pandas as pd
from am_combiner.utils.spectral import get_node_eign_vector, get_graph_eign, get_graph_clusters
from am_combiner.features.common import Article, Features

ARTICLE_A = Article("Johny", "Johny is a bad guy", "abc1.com")
ARTICLE_B = Article("Johny", "Johny meet Joseph and Chan", "abc2.com")
ARTICLE_A.extracted_entities[Features.PERSON_CLEAN] = ["Johny Cash", "Joseph"]
ARTICLE_B.extracted_entities[Features.PERSON_CLEAN] = ["Chan"]
input_articles = [ARTICLE_A, ARTICLE_B]
use_features = [Features.PERSON_CLEAN]


def test_graph_eign():
    graph_obj, eign_val, eign_vectr = get_graph_eign(input_articles, use_features)

    assert list(graph_obj.nodes) == ["abc1.com", "Johny", "abc2.com"]
    assert np.alltrue(
        eign_val == np.array([-3.367702055640532e-17, 0.9999999999999998, 2.999999999999999])
    )
    assert eign_vectr.shape == (3, 3)

    return graph_obj, eign_vectr


def test_node_eign_vector():
    graph_obj, eign_vectr = test_graph_eign()

    graph_df = get_node_eign_vector(graph_obj, input_articles, eign_vectr)

    assert graph_df.equals(
        pd.DataFrame(
            {
                "node_val": ["abc1.com", "abc2.com"],
                "vector": [
                    [-0.7071067811865474, -0.4082482904638627],
                    [0.7071067811865476, -0.4082482904638632],
                ],
            }
        )
    )

    return graph_df


def test_graph_clusters():
    graph_df = test_node_eign_vector()

    sim, graphs, sub_graphs = get_graph_clusters(graph_df, 0.4)

    assert np.alltrue(
        sim
        == np.array(
            [[1.0000000000000002, -0.5000000000000001], [-0.5000000000000001, 0.9999999999999998]]
        )
    )
    assert list(graphs.nodes) == [0, 1]
    # counting number of subgraphs
    assert len(list(sub_graphs)) == 2
