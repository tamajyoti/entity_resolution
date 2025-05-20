from typing import List, Dict

from am_combiner.features.article import Features, Article
from am_combiner.features.common import ArticleVisitor
import pandas as pd


class GraphDataVisitor(ArticleVisitor):

    """
    Class is a concrete implementation of a visitor pattern.

    Attributes
    ----------
        use_features: List[Feature]
            A list of features used to build a graph representation.

    """

    def __init__(self, use_features: List[Features]):
        super().__init__()
        self.use_features = use_features

    def visit_article(self, article: Article) -> None:
        """
        Concrete implementation of the abstract method.

        Parameters
        ----------
        article: Article
            An article to be visited

        Returns
        -------
        None

        """
        nodes, edges = GraphVisualizationDataBuilder.get_single_article_graph_repr(
            article, self.use_features
        )
        article.extracted_entities[Features.WEB_GRAPH_VIS] = {"nodes": nodes, "links": edges}


class GraphVisualizationDataBuilder:

    """
    A namespace to keep static methods for graph visualization creation.

    Contains two static methods for
    1. representing a single articles into a graph
    2. combining a set of individual representations into one single representation

    """

    @staticmethod
    def get_single_article_graph_repr(article: Article, features: List[Features]):
        """
        Create a graph representation for a single article.

        Parameters
        ----------
        article: Article
            A source article for representation creation
        features: List[Feature]
            A list of features used for graph representation

        Returns
        -------
        Tuple[List[Dict], List[Dict]]
            A set of nodes and links between them

        """
        all_nodes: List[Dict] = []
        all_edges: List[Dict] = []
        for feature_id, feature in enumerate(features):
            extracted_entities = article.extracted_entities[feature]
            for ent in extracted_entities:
                ent_str = str(ent).strip()

                node_dict = {"id": ent_str, "group": feature_id}
                edge_dict = {"source": article.url, "target": ent_str, "value": feature_id}

                all_nodes.append(node_dict)
                all_edges.append(edge_dict)

        # This is make sure that the entity node(article url that is) is displayed
        all_nodes.append({"id": article.url, "group": len(features)})
        return all_nodes, all_edges

    @staticmethod
    def merge_article_graph_representations(
        articles: List[Article], from_feature: Features = Features.WEB_GRAPH_VIS
    ):
        """
        Convert individual article's graph representations to a combined graph representation.

        Parameters
        ----------
        from_feature: Features
            The field name where to take the graph data from
        articles: List[Article]
            A list of source articles.

        Returns
        -------
        Dict
            A dictionary with two fields: nodes and links representing the graph structure.

        """
        all_nodes = []
        all_links = []

        for article in articles:
            nodes = article.extracted_entities[from_feature]["nodes"]
            links = article.extracted_entities[from_feature]["links"]
            all_nodes.extend(nodes)
            all_links.extend(links)

        all_nodes = pd.DataFrame(all_nodes).drop_duplicates(["id"]).to_dict("records")
        all_links = pd.DataFrame(all_links).drop_duplicates(["source", "target"]).to_dict("records")

        graph_data = {"nodes": all_nodes, "links": all_links}
        return graph_data
