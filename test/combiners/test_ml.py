import dgl
import torch

from am_combiner.combiners.ml import GCN, HeteroGCN, GCNCombiner
import networkx as nx
import numpy as np
from am_combiner.features.nn.common import HETEROGENEOUS_NODE_NAME


class TestMLComponents:
    def test_simple_gcn_implementation(self):
        gcn = GCN(in_feats=8000, rep_dim=128)
        assert gcn.conv1.fc_self.in_features == 8000
        assert gcn.conv1.fc_neigh.in_features == 8000
        assert gcn.conv1.fc_self.out_features == 128
        assert gcn.conv1.fc_neigh.out_features == 128

    def test_simple_gcn_returns_embedding_of_correct_size(self):
        Gnx = nx.DiGraph(np.array([[0, 1], [1, 0]]))
        G = dgl.from_networkx(Gnx)

        gcn = GCN(in_feats=10, rep_dim=128)

        embedding = torch.nn.Embedding(num_embeddings=2, embedding_dim=10)
        out = gcn(G, embedding.weight, None)
        assert out.shape[0] == 2
        assert out.shape[1] == 128


class TestHeteroComponents:
    def test_mods_creation(self):
        hge = HeteroGCN(in_feats=15, mods=["a", "b"], rep_dim=10)

        assert hge.conv.mods["a"].fc_self.in_features == 15
        assert hge.conv.mods["a"].fc_self.out_features == 10
        assert sorted(hge.conv.mods.keys()) == ["a", "b"]

    def test_hetero_edge_weights(self):

        hge = HeteroGCN(in_feats=15, mods=["a", "b"], rep_dim=10)

        multigaph_source = {}
        for mod in ["a", "b"]:
            triplet = (HETEROGENEOUS_NODE_NAME, mod, HETEROGENEOUS_NODE_NAME)
            multigaph_source[triplet] = (torch.tensor([0, 1]), torch.tensor([1, 0]))

        G = dgl.heterograph(multigaph_source, num_nodes_dict={HETEROGENEOUS_NODE_NAME: 2})
        embedding = {HETEROGENEOUS_NODE_NAME: torch.ones((2, 15))}

        for mod in ["a", "b"]:
            G.edges[mod].data["weight"] = torch.Tensor([2.0, 3.0])
        edge_weights = GCNCombiner.get_edge_weight(G)
        for mod in ["a", "b"]:
            G.edges[mod].data["weight"] = torch.ones(2)
        unit_edge_weights = GCNCombiner.get_edge_weight(G)

        edge_weight_output = hge(G, embedding, edge_weights).detach().numpy()
        unit_edge_weight_output = hge(G, embedding, unit_edge_weights).detach().numpy()

        assert (unit_edge_weight_output != edge_weight_output).any()
