"""MOHITO baseline adapter for zero-shot cross-domain evaluation.

This module adapts our EventDrivenEnv to MOHITO's expected hypergraph format,
enabling zero-shot evaluation using pretrained MOHITO actors from the 
rideshare domain.

Reference:
    Anil, G., Doshi, P., Redder, D., Eck, A., & Soh, L. K. (2025).
    MOHITO: Multi-Agent Reinforcement Learning using Hypergraphs for 
    Task-Open Systems. UAI 2025.

Key Design Decisions:
    1. Multi-vehicle parameter sharing: All vehicles share one actor network
    2. Minimal hypergraph: Structural contract alignment, not semantic equivalence
    3. Heuristic feature mapping: Our features mapped to MOHITO schema
    4. Zero exploration: epsilon=0 for deterministic reproducibility
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Lazy imports to avoid torch_geometric dependency when not using this policy
Data = None
add_self_loops = None
ActorNetwork = None


def _ensure_pyg_imports():
    """Lazily import torch_geometric components."""
    global Data, add_self_loops
    if Data is not None:
        return
    
    from torch_geometric.data import Data as _Data
    from torch_geometric.utils import add_self_loops as _add_self_loops
    Data = _Data
    add_self_loops = _add_self_loops


def _ensure_mohito_imports():
    """Lazily import MOHITO ActorNetwork from baselines."""
    global ActorNetwork
    if ActorNetwork is not None:
        return
    
    mohito_path = Path(__file__).parent.parent.parent / "baselines" / "mohito-public" / "rideshare"
    if str(mohito_path) not in sys.path:
        sys.path.insert(0, str(mohito_path))
    
    # Import GAT-based actor from MOHITO rideshare
    from mohitoR.gat import ActorNetwork as _ActorNetwork
    ActorNetwork = _ActorNetwork


# Node type constants (matching MOHITO rideshare)
NODE_TYPE_EDGE = 0
NODE_TYPE_AGENT = 1
NODE_TYPE_TASK = 2
NODE_TYPE_ACTION = 3


def build_mohito_graph(
    env: Any,
    features: Dict[str, np.ndarray],
    vehicle_idx: int,
    grid_size: int = 10,
    mode: str = "compact",
) -> Tuple[Data, List[int], List[List[float]]]:
    """Construct minimal hypergraph meeting MOHITO's structural contract.
    
    This builds the minimal graph structure that allows MOHITO's ActorNetwork
    to run forward() without errors. We do NOT attempt semantic equivalence
    with the rideshare domain - this is a zero-shot baseline.
    
    Args:
        env: EventDrivenEnv instance
        features: Feature dict from env.get_feature_batch()
        vehicle_idx: Current vehicle index (for multi-agent)
        grid_size: Virtual grid size for location encoding
        
    Returns:
        graph_data: PyG Data object with MOHITO-compatible structure
        edge_space: List of edge node indices (action candidates)
        action_space: List of action feature vectors
    """
    _ensure_pyg_imports()

    mode = str(mode).lower().strip()
    if mode not in {"compact", "full"}:
        raise ValueError(f"Invalid mohito graph mode: {mode} (expected compact/full)")

    if mode == "compact":
        actions = features["actions"].astype(np.int64)
        node_features = features["node_features"]
        edge_features = features["edge_features"]
        current_node_idx = int(features["current_node_index"][0])

        num_actions = int(len(actions))
        num_vehicles = int(getattr(env, "num_vehicles", getattr(getattr(env, "config", None), "num_vehicles", 1)))

        node_list: List[List[float]] = []

        # 1) Agent nodes (one per vehicle; lightweight state)
        for v_idx in range(num_vehicles):
            onboard_count = 0
            accepted_count = 0
            location = 0
            try:
                vehicle = env.vehicles[v_idx]
                onboard_count = len(getattr(vehicle, "onboard", []))
                accepted_count = len(getattr(vehicle, "accepted", []))
                location = int(getattr(vehicle, "current_stop", current_node_idx)) % grid_size
            except Exception:
                pass
            node_list.append(
                [NODE_TYPE_AGENT, float(v_idx), float(location), float(accepted_count), float(onboard_count)]
            )

        # 2) Action nodes (candidate destinations)
        action_nodes_start_idx = len(node_list)
        action_space: List[List[float]] = []
        start_loc = current_node_idx % (grid_size * grid_size)
        for i, stop_id in enumerate(actions):
            travel_time = 0.0
            if i < len(edge_features):
                ef = edge_features[i]
                travel_time = float(ef[3]) if len(ef) > 3 else 0.0
            end_loc = int(stop_id) % (grid_size * grid_size)
            # Node features for the GNN (5 dims)
            node_list.append([NODE_TYPE_ACTION, float(start_loc), float(end_loc), float(travel_time), 0.0])
            # Action-space vector used by mohito_policy for mapping (keeps end_loc in index 2)
            action_space.append([NODE_TYPE_ACTION, float(start_loc), float(end_loc), 0.0, 0.0])
        num_actions_with_noop = num_actions

        # 3) Edge nodes: one per action (including NOOP)
        edge_nodes_start_idx = len(node_list)
        for _ in range(num_actions_with_noop):
            node_list.append(
                [NODE_TYPE_EDGE]
                + [float(np.random.uniform(1e-5, 1e-4)) for _ in range(4)]
            )

        # 4) Build adjacency: active agent <-> edge_node <-> action_node
        edges_src: List[int] = []
        edges_dst: List[int] = []
        active_agent_idx = int(max(0, min(vehicle_idx, num_vehicles - 1))) if num_vehicles > 0 else 0
        for k in range(num_actions_with_noop):
            e_idx = edge_nodes_start_idx + k
            a_idx = action_nodes_start_idx + k
            edges_src.extend([active_agent_idx, e_idx, e_idx, a_idx])
            edges_dst.extend([e_idx, active_agent_idx, a_idx, e_idx])

        feature_matrix = np.array(node_list, dtype=np.float32)
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_index, _ = add_self_loops(edge_index, num_nodes=len(node_list))
        x = torch.from_numpy(feature_matrix).float()
        data = Data(x=x, edge_index=edge_index)
        edge_space = [edge_nodes_start_idx + k for k in range(num_actions_with_noop)]
        return data, edge_space, action_space
    
    actions = features["actions"].astype(np.int64)
    action_mask = features["action_mask"].astype(bool)
    node_features = features["node_features"]
    edge_features = features["edge_features"]
    current_node_idx = int(features["current_node_index"][0])
    
    num_actions = len(actions)
    num_vehicles = getattr(env, 'num_vehicles', 1)
    
    # Build node feature matrix
    # Format: [node_type, field1, field2, field3, field4]
    
    node_list = []
    node_names = []
    
    # 1. Agent nodes (one per vehicle)
    vehicle = env._get_active_vehicle() if hasattr(env, '_get_active_vehicle') else None
    for v_idx in range(num_vehicles):
        if v_idx == vehicle_idx and vehicle is not None:
            # Current vehicle - use actual state
            onboard_count = len(vehicle.onboard)
            accepted_count = len(getattr(vehicle, 'accepted', []))
            location = current_node_idx % grid_size
        else:
            # Other vehicles - use placeholder
            onboard_count = 0
            accepted_count = 0
            location = 0
        
        agent_node = [NODE_TYPE_AGENT, v_idx, location, accepted_count, onboard_count]
        node_list.append(agent_node)
        node_names.append(f"ag{v_idx}")
    
    # 2. Task nodes (waiting passengers at stops with queue)
    # Heuristic: use stops with non-zero queue as "unassigned tasks"
    waiting = getattr(env, 'waiting', {})
    task_nodes_start_idx = len(node_list)
    task_count = 0
    task_to_stop = {}  # Map task index to stop ID
    
    for stop_id, queue in waiting.items():
        if queue and len(queue) > 0:
            # Create task node for this stop
            # Heuristic mapping: [type, start_loc, end_loc, agent_assigned, agent_riding]
            start_loc = stop_id % (grid_size * grid_size)
            # Use first passenger's destination if available
            first_pax = queue[0]
            end_loc = first_pax.get("dropoff_stop_id", start_loc) % (grid_size * grid_size)
            
            task_node = [NODE_TYPE_TASK, start_loc, end_loc, -1, -1]
            node_list.append(task_node)
            node_names.append(f"ct{task_count}")
            task_to_stop[task_count] = stop_id
            task_count += 1
    
    # Add NOOP task
    noop_task_node = [NODE_TYPE_TASK, -1, -1, -1, -1]
    node_list.append(noop_task_node)
    node_names.append(f"ct{task_count}")
    task_count += 1
    
    # 3. Action nodes (candidate destinations)
    action_nodes_start_idx = len(node_list)
    action_space = []
    
    for i, stop_id in enumerate(actions):
        if i < len(edge_features):
            ef = edge_features[i]
            travel_time = float(ef[3]) if len(ef) > 3 else 0.0
        else:
            travel_time = 0.0
        
        # Heuristic: action node features
        start_loc = current_node_idx % (grid_size * grid_size)
        end_loc = stop_id % (grid_size * grid_size)
        action_type = 0  # "accept" type for new tasks
        entry_step = 0
        
        action_node = [NODE_TYPE_ACTION, start_loc, end_loc, action_type, entry_step]
        node_list.append(action_node)
        node_names.append(f"ca{i}")
        action_space.append(action_node)
    
    # 4. Edge nodes (hyperedge connectors)
    # Connect agent -> task -> action for each candidate
    edge_nodes_start_idx = len(node_list)
    num_edge_nodes = num_vehicles * (task_count)  # All agents connected to all tasks
    
    for i in range(num_edge_nodes):
        # Small random values to avoid zero gradients
        edge_node = [NODE_TYPE_EDGE] + [np.random.uniform(1e-5, 1e-4) for _ in range(4)]
        node_list.append(edge_node)
        node_names.append(f"e{i}")
    
    # Build adjacency (edge_index)
    # MOHITO expects hyperedges connecting: agent <-> edge_node <-> task <-> action
    edges_src = []
    edges_dst = []
    edge_space = []
    
    e_idx = 0
    for ag_idx in range(num_vehicles):
        for t_idx in range(task_count):
            e_node_idx = edge_nodes_start_idx + e_idx
            ag_node_idx = ag_idx
            task_node_idx = task_nodes_start_idx + t_idx
            action_node_idx = action_nodes_start_idx + min(t_idx, max(0, num_actions - 1))
            
            # Bidirectional edges
            edges_src.extend([e_node_idx, ag_node_idx, e_node_idx, task_node_idx, e_node_idx, action_node_idx])
            edges_dst.extend([ag_node_idx, e_node_idx, task_node_idx, e_node_idx, action_node_idx, e_node_idx])
            
            # Track edge nodes for current agent
            if ag_idx == vehicle_idx:
                edge_space.append(e_node_idx)
            
            e_idx += 1
    
    # Convert to tensors
    feature_matrix = np.array(node_list, dtype=np.float32)
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_index, _ = add_self_loops(edge_index, num_nodes=len(node_list))
    
    x = torch.from_numpy(feature_matrix).float()
    data = Data(x=x, edge_index=edge_index)
    
    return data, edge_space, action_space


def load_mohito_actor(
    model_path: str,
    config: Dict[str, Any],
    device: torch.device,
) -> Any:
    """Load MOHITO actor network.
    
    Supports multiple checkpoint formats:
    1. Unified checkpoint (dict with "model_state_dict" key) - from in-domain training
    2. Direct state_dict - legacy/zero-shot format
    
    Args:
        model_path: Path to saved actor state dict (.pth)
        config: MOHITO configuration dict
        device: Torch device
        
    Returns:
        Loaded ActorNetwork in eval mode
    """
    _ensure_mohito_imports()
    
    feature_len = int(config.get("feature_len", 5))
    num_layers = int(config.get("num_layers_actor", 20))
    hidden_dim = int(config.get("hidden_dim", 50))
    heads = int(config.get("heads", 2))
    lr = float(config.get("lr_actor", 0.001))
    beta = float(config.get("beta", 0.001))
    
    actor = ActorNetwork(
        num_state_features=feature_len,
        LR_A=lr,
        BETA=beta,
        hidden_dim_actor=hidden_dim,
        num_layers=num_layers,
        heads=heads,
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if this is a unified checkpoint or direct state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # Unified checkpoint format from in-domain training
        actor.main.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Legacy format: direct state_dict (zero-shot weights)
        actor.main.load_state_dict(checkpoint)
    
    actor.to(device)
    actor.eval()
    
    return actor


def mohito_policy(
    env: Any,
    features: Dict[str, np.ndarray],
    actor: Any,
    config: Dict[str, Any],
    device: torch.device,
) -> Optional[int]:
    """MOHITO zero-shot policy for evaluation.
    
    This is a deterministic policy (epsilon=0) that:
    1. Builds minimal hypergraph from current state
    2. Runs forward pass through pretrained actor
    3. Returns selected stop ID respecting action mask
    
    Args:
        env: EventDrivenEnv instance
        features: Feature dict from env.get_feature_batch()
        actor: Loaded ActorNetwork
        config: MOHITO configuration
        device: Torch device
        
    Returns:
        Selected stop ID or None if no valid action
    """
    actions = features["actions"].astype(np.int64)
    action_mask = features["action_mask"].astype(bool)
    
    if len(actions) == 0:
        return None
    
    valid_indices = np.where(action_mask)[0]
    if len(valid_indices) == 0:
        return None
    
    # Get current vehicle index (for multi-agent parameter sharing)
    vehicle = env._get_active_vehicle() if hasattr(env, '_get_active_vehicle') else None
    vehicle_idx = getattr(vehicle, 'vehicle_id', 0) if vehicle else 0
    
    grid_size = int(config.get("grid_size", 10))
    graph_mode = str(config.get("graph_mode", "compact"))
     
    # Build MOHITO-compatible graph
    graph_data, edge_space, action_space = build_mohito_graph(
        env, features, vehicle_idx, grid_size, mode=graph_mode
    )
    
    # Move to device
    graph_data = graph_data.to(device)
    edge_space_tensor = torch.tensor(edge_space, dtype=torch.long, device=device)
    action_space_tensor = torch.tensor(action_space, dtype=torch.float32, device=device)
    
    # Forward pass (no exploration, epsilon=0)
    with torch.no_grad():
        try:
            edge_value, selected_action = actor.getAction(
                graph_data,
                edge_space_tensor,
                action_space_tensor,
                network='main',
                training=False,
            )
        except Exception as e:
            # Fallback to first valid action if actor fails
            return int(actions[valid_indices[0]])
    
    # Map MOHITO action back to stop ID
    # selected_action is an action feature vector, find matching action index
    if isinstance(selected_action, torch.Tensor):
        selected_action = selected_action.cpu().numpy()
    
    # Find closest matching action by index
    action_idx = 0
    if len(selected_action) > 1:
        # Use start/end location to match
        target_end = int(selected_action[2]) if len(selected_action) > 2 else 0
        for i, stop_id in enumerate(actions):
            if action_mask[i] and stop_id % 100 == target_end:
                action_idx = i
                break
        else:
            # No exact match, use first valid
            action_idx = int(valid_indices[0])
    else:
        action_idx = int(min(selected_action[0], len(actions) - 1))
    
    # Ensure action respects mask (critical requirement)
    if not action_mask[action_idx]:
        action_idx = int(valid_indices[0])
    
    return int(actions[action_idx])
