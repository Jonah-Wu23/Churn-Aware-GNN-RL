# Model Specification (Edge-Q GNN)

## Architecture
- Edge-Conditioned Convolution (ECC) implemented in pure PyTorch
- Action Q-head scores candidate edges using [h_src, h_dst, edge_attr]

## Inputs
- Node features: risk_mean, risk_cvar, wait_count, fairness_weight, geo_embedding (scalar `emb_geo_0`)
- Edge features: delta_eta_max, delta_cvar_onboard, violation_count, travel_time

## Data Contract (Current Implementation)
- node_features: [num_nodes, node_feat_dim]
- graph_edge_index: [2, num_graph_edges] for ECC message passing
- graph_edge_features: [num_graph_edges, edge_feat_dim] (travel_time stored at index 3, others 0)
- action_edge_index: [2, num_action_edges] candidate edges from current stop
- edge_features: [num_action_edges, edge_feat_dim]
- action_mask: [num_action_edges] provided by the env

## Outputs
- Edge Q values aligned with action_edge_index / edge_features

## Inference Constraints
- Target latency < 10 ms per vehicle decision
- k-hop subgraph extraction for local inference

## TODO
- Provide optional GAT variant and config switch
- Define maximum batch size for inference latency tests
