import numpy as np
from neo4j import GraphDatabase

from am_combiner.features.common import SanctionVisitor
from am_combiner.features.sanction import Sanction, SanctionFeatures


class Neo4jEmbeddingVisitor(SanctionVisitor):

    """Node2Vec embeddings downloader."""

    def __init__(
        self,
        feature_name: str = "rp",
        uri: str = "neo4j://neo4j:7687",
        username: str = "neo4j",
        password: str = "neo4j",
    ):
        super().__init__()
        driver = GraphDatabase.driver(uri, auth=(username, password))
        session = driver.session()

        res = session.run("match (e:SANCTION_ID) return e")
        cache = {}
        print(f"Downloading {feature_name} embeddings...")
        for ind, r in enumerate(res):
            sid = r.get("e")["value"]
            feature = np.array(r.get("e")[feature_name])
            cache[sid] = feature
        print(f"Done downloading {ind} embeddings")
        self.cache = cache

    def visit_sanction(self, sanction: Sanction) -> None:
        """Download node2vec embeddings."""
        sanction.extracted_entities[SanctionFeatures.REMOTE] = self.cache[sanction.sanction_id]
