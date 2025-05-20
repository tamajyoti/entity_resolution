import pytest

from am_combiner.features.article import Features, Article
from am_combiner.features.graph_data import GraphVisualizationDataBuilder, GraphDataVisitor


class TestGraphDataProducers:
    @pytest.mark.parametrize(
        ["article_dict", "extracted_entities", "expected_nodes", "expected_links"],
        [
            (
                {
                    "entity_name": "Johny",
                    "article_text": "Johny is a bad guy",
                    "url": "http://lol.com",
                },
                {
                    Features.PERSON_CLEAN: ["Johny Cash", "Joseph"],
                    Features.ORG_CLEAN: ["American Tower"],
                },
                [
                    {"id": "Johny Cash", "group": 0},
                    {"id": "Joseph", "group": 0},
                    {"id": "American Tower", "group": 1},
                    {"id": "http://lol.com", "group": 2},
                ],
                [
                    {"source": "http://lol.com", "target": "Johny Cash", "value": 0},
                    {"source": "http://lol.com", "target": "Joseph", "value": 0},
                    {"source": "http://lol.com", "target": "American Tower", "value": 1},
                ],
            ),
        ],
    )
    def test_single_article_graph_building(
        self, article_dict, extracted_entities, expected_nodes, expected_links
    ):
        article_for_graph = Article(**article_dict)
        article_for_graph.extracted_entities = extracted_entities
        nodes, links = GraphVisualizationDataBuilder.get_single_article_graph_repr(
            article_for_graph, features=[Features.PERSON_CLEAN, Features.ORG_CLEAN]
        )
        assert nodes == expected_nodes
        assert links == expected_links

    @pytest.mark.parametrize(
        ["extracted_entities_1", "extracted_entities_2", "expected_nodes", "expected_links"],
        [
            (
                {
                    Features.PERSON_CLEAN: ["Johny Cash", "Joseph"],
                    Features.ORG_CLEAN: ["American Tower"],
                },
                {
                    Features.PERSON_CLEAN: ["TCash", "JJoseph"],
                    Features.ORG_CLEAN: ["AAmerican Tower"],
                },
                [
                    {"id": "Johny Cash", "group": 0},
                    {"id": "Joseph", "group": 0},
                    {"id": "American Tower", "group": 1},
                    {"id": "url1", "group": 2},
                    {"id": "TCash", "group": 0},
                    {"id": "JJoseph", "group": 0},
                    {"id": "AAmerican Tower", "group": 1},
                    {"id": "url2", "group": 2},
                ],
                [
                    {"source": "url1", "target": "Johny Cash", "value": 0},
                    {"source": "url1", "target": "Joseph", "value": 0},
                    {"source": "url1", "target": "American Tower", "value": 1},
                    {"source": "url2", "target": "TCash", "value": 0},
                    {"source": "url2", "target": "JJoseph", "value": 0},
                    {"source": "url2", "target": "AAmerican Tower", "value": 1},
                ],
            ),
            (
                {
                    Features.PERSON_CLEAN: ["Johny Cash", "Joseph"],
                    Features.ORG_CLEAN: ["American Tower"],
                },
                {
                    Features.PERSON_CLEAN: ["Johny Cash", "JJoseph"],
                    Features.ORG_CLEAN: ["AAmerican Tower"],
                },
                [
                    {"id": "Johny Cash", "group": 0},
                    {"id": "Joseph", "group": 0},
                    {"id": "American Tower", "group": 1},
                    {"id": "url1", "group": 2},
                    {"id": "JJoseph", "group": 0},
                    {"id": "AAmerican Tower", "group": 1},
                    {"id": "url2", "group": 2},
                ],
                [
                    {"source": "url1", "target": "Johny Cash", "value": 0},
                    {"source": "url1", "target": "Joseph", "value": 0},
                    {"source": "url1", "target": "American Tower", "value": 1},
                    {"source": "url2", "target": "Johny Cash", "value": 0},
                    {"source": "url2", "target": "JJoseph", "value": 0},
                    {"source": "url2", "target": "AAmerican Tower", "value": 1},
                ],
            ),
        ],
    )
    def test_two_articles_merged(
        self, extracted_entities_1, extracted_entities_2, expected_nodes, expected_links
    ):
        article1 = Article(entity_name="", article_text="", url="url1")
        article1.extracted_entities = extracted_entities_1
        article2 = Article(entity_name="", article_text="", url="url2")
        article2.extracted_entities = extracted_entities_2

        v = GraphDataVisitor(use_features=[Features.PERSON_CLEAN, Features.ORG_CLEAN])
        article1.accept_visitor(v)
        article2.accept_visitor(v)

        graph_data = GraphVisualizationDataBuilder.merge_article_graph_representations(
            [article1, article2]
        )
        assert graph_data["nodes"] == expected_nodes
        assert graph_data["links"] == expected_links
