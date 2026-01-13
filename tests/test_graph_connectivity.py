import networkx as nx
import pandas as pd


def test_layer2_connectivity():
    nodes = pd.read_parquet("data/processed/graph/layer2_nodes.parquet")
    edges = pd.read_parquet("data/processed/graph/layer2_edges.parquet")

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes["gnn_node_id"].astype(int).tolist())
    graph.add_edges_from(edges[["source", "target"]].astype(int).itertuples(index=False, name=None))

    in_deg = dict(graph.in_degree())
    out_deg = dict(graph.out_degree())
    isolated = [n for n in graph.nodes if in_deg.get(n, 0) == 0 and out_deg.get(n, 0) == 0]

    assert len(isolated) == 0, f"Found isolated nodes: {isolated[:10]}"
    assert nx.is_weakly_connected(graph), "Layer2 graph is not weakly connected"
