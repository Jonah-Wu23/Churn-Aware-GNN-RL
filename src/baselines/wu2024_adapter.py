"""Wu et al. (C&IE 2024) baseline adapter for architecture placeholder evaluation.

This module implements a zero-shot cross-domain adapter for the Wu et al.
"Multi-Agent Deep Reinforcement Learning based Real-time Planning Approach
for Responsive Customized Bus Routes" paper.

Reference:
    Wu, X. et al. (2024). Multi-Agent Deep Reinforcement Learning based
    Real-time Planning Approach for Responsive Customized Bus Routes.
    Computers & Industrial Engineering, 186, 109764.

=== CRITICAL DECLARATIONS (READ BEFORE USE) ===

1. WEIGHT STATUS (pretrained=false):
   The baseline repository contains NO pretrained .pt files.
   This implementation uses random_init or uniform_logits mode.
   Results do NOT represent the paper's method performance ceiling.

2. DOMAIN TRANSFER:
   EventDrivenEnv uses dynamic k-hop subgraph, not fixed station topology.
   This implementation uses heuristic feature mapping (Strategy S1).
   No domain retraining is performed.

3. REPRODUCIBILITY:
   All randomness is controlled by seed.
   Evaluation JSON records: seed, config_hash, weights_mode.

=== DESIGN: Strategy S1 (Fixed Kmax + Padding) ===

Each decision step:
1. Get candidate actions from env (already filtered by action_mask)
2. Build candidate station list: [current] + top-(Kmax-1) candidates
3. Pad to Kmax if fewer candidates; truncate if more
4. Apply mask: dummy positions and infeasible actions masked = 0

This provides constant input dimension for PointerNet-style model.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Model Architecture (simplified from original)
# =============================================================================

class CB_Encoder(nn.Module):
    """1D-Conv encoder for static/dynamic features (Wu et al. Section 3.2)."""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)  # (batch, hidden_size, seq_len)


class CB_Attention(nn.Module):
    """Bahdanau-style attention for context computation."""
    
    def __init__(self, hidden_size: int, device: torch.device):
        super().__init__()
        self.v = nn.Parameter(torch.randn(1, 1, hidden_size, device=device) * 0.01)
        self.W = nn.Parameter(torch.randn(1, hidden_size, 3 * hidden_size, device=device) * 0.01)
    
    def forward(
        self, 
        static_hidden: torch.Tensor,
        dynamic_hidden: torch.Tensor,
        decoder_hidden: torch.Tensor
    ) -> torch.Tensor:
        batch_size, hidden_size, seq_len = static_hidden.size()
        
        # Expand decoder hidden to match
        hidden = decoder_hidden.unsqueeze(2).expand_as(static_hidden)
        hidden = torch.cat((static_hidden, dynamic_hidden, hidden), 1)
        
        v = self.v.expand(batch_size, 1, hidden_size)
        W = self.W.expand(batch_size, hidden_size, -1)
        
        attns = torch.bmm(v, torch.tanh(torch.bmm(W, hidden)))
        attns = F.softmax(attns, dim=2)
        return attns


class CB_Agent(nn.Module):
    """GRU + Attention decoder for action selection (Wu et al. Section 3.3)."""
    
    def __init__(self, hidden_size: int, num_layers: int = 1, dropout: float = 0.1, device: torch.device = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device or torch.device('cpu')
        
        self.v = nn.Parameter(torch.randn(1, 1, hidden_size, device=self.device) * 0.01)
        self.W = nn.Parameter(torch.randn(1, hidden_size, 2 * hidden_size, device=self.device) * 0.01)
        
        self.gru = nn.GRU(
            hidden_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.encoder_attn = CB_Attention(hidden_size, self.device)
        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)
    
    def forward(
        self,
        static_hidden: torch.Tensor,
        dynamic_hidden: torch.Tensor,
        decoder_hidden: torch.Tensor,
        last_hh: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rnn_out, last_hh = self.gru(decoder_hidden.transpose(2, 1), last_hh)
        rnn_out = rnn_out.squeeze(1)
        
        rnn_out = self.drop_rnn(rnn_out)
        if self.num_layers == 1:
            last_hh = self.drop_hh(last_hh)
        
        enc_attn = self.encoder_attn(static_hidden, dynamic_hidden, rnn_out)
        context = enc_attn.bmm(static_hidden.permute(0, 2, 1))
        
        context = context.transpose(1, 2).expand_as(static_hidden)
        energy = torch.cat((static_hidden, context), dim=1)
        
        v = self.v.expand(static_hidden.size(0), -1, -1)
        W = self.W.expand(static_hidden.size(0), -1, -1)
        
        probs = torch.bmm(v, torch.tanh(torch.bmm(W, energy))).squeeze(1)
        return probs, last_hh


class Wu2024PointerNet(nn.Module):
    """Simplified PointerNetwork for single-agent stop selection.
    
    This is a simplified version focusing on the core architecture:
    - Static encoder: encodes station features
    - Dynamic encoder: encodes current state features
    - Decoder: GRU + Attention for action selection
    
    Original paper uses 3 separate agents; we use parameter sharing.
    """
    
    def __init__(
        self,
        static_size: int,
        dynamic_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        device: torch.device = None,
    ):
        super().__init__()
        self.device = device or torch.device('cpu')
        
        self.static_encoder = CB_Encoder(static_size, hidden_size)
        self.dynamic_encoder = CB_Encoder(dynamic_size, hidden_size)
        self.decoder = CB_Encoder(static_size, hidden_size)
        self.pointer = CB_Agent(hidden_size, num_layers, dropout, self.device)
        
        self.x0 = nn.Parameter(torch.zeros(1, static_size, 1, device=self.device))
        
        # Xavier initialization
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        static: torch.Tensor,
        dynamic: torch.Tensor,
        mask: torch.Tensor,
        decoder_input: Optional[torch.Tensor] = None,
        last_hh: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single-step forward pass.
        
        Args:
            static: Static features [batch, static_size, num_stops]
            dynamic: Dynamic features [batch, dynamic_size, num_stops]
            mask: Valid action mask [batch, num_stops], 1=valid, 0=masked
            decoder_input: Previous selection [batch, static_size, 1]
            last_hh: Previous GRU hidden state
            
        Returns:
            probs: Action probabilities [batch, num_stops]
        """
        batch_size = static.size(0)
        
        if decoder_input is None:
            decoder_input = self.x0.expand(batch_size, -1, -1)
        
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)
        decoder_hidden = self.decoder(decoder_input)
        
        logits, _ = self.pointer(static_hidden, dynamic_hidden, decoder_hidden, last_hh)
        
        # Apply mask (log-mask for softmax stability)
        masked_logits = logits + mask.log().clamp(min=-1e9)
        probs = F.softmax(masked_logits, dim=1)
        
        return probs


# =============================================================================
# Feature Mapping (V0: Minimal Closed-Loop)
# =============================================================================

def _build_static_features(
    env: Any,
    features: Dict[str, np.ndarray],
    kmax: int,
    candidate_stops: List[int],
) -> np.ndarray:
    """Build static feature tensor for Wu2024 model.
    
    V0 Features (per candidate stop):
    - X coordinate (normalized)
    - Y coordinate (normalized)
    - Travel time from current position
    
    Returns:
        static: [1, 3, kmax] tensor
    """
    node_features = features["node_features"]
    edge_features = features["edge_features"]
    actions = features["actions"]
    current_idx = int(features["current_node_index"][0])
    
    static = np.zeros((3, kmax), dtype=np.float32)
    
    # Get stop coordinates from env
    stop_coords = getattr(env, 'stop_coords', {})
    
    # Get coordinate bounds for normalization
    if stop_coords:
        all_lons = [c[0] for c in stop_coords.values()]
        all_lats = [c[1] for c in stop_coords.values()]
        lon_range = max(all_lons) - min(all_lons) + 1e-6
        lat_range = max(all_lats) - min(all_lats) + 1e-6
        lon_min, lat_min = min(all_lons), min(all_lats)
    else:
        lon_range = lat_range = 1.0
        lon_min = lat_min = 0.0
    
    for i, stop_id in enumerate(candidate_stops):
        if stop_id < 0:  # padding
            continue
        
        # Coordinates
        if stop_id in stop_coords:
            lon, lat = stop_coords[stop_id]
            static[0, i] = (lon - lon_min) / lon_range
            static[1, i] = (lat - lat_min) / lat_range
        
        # Travel time (from edge features if available)
        if i < len(edge_features):
            static[2, i] = edge_features[i][3] / 600.0  # Normalize to ~[0,1]
    
    return static.reshape(1, 3, kmax)


def _build_dynamic_features(
    env: Any,
    features: Dict[str, np.ndarray],
    kmax: int,
    candidate_stops: List[int],
) -> np.ndarray:
    """Build dynamic feature tensor for Wu2024 model.
    
    V0 Features (per candidate stop):
    - Waiting count (normalized)
    - Vehicle load (broadcast)
    - Current time (broadcast, normalized)
    
    Returns:
        dynamic: [1, 3, kmax] tensor
    """
    dynamic = np.zeros((3, kmax), dtype=np.float32)
    
    waiting = getattr(env, 'waiting', {})
    current_time = getattr(env, 'current_time', 0.0)
    max_time = 3600.0 * 2  # 2 hours normalize factor
    
    # Get vehicle load
    vehicle = env._get_active_vehicle() if hasattr(env, '_get_active_vehicle') else None
    vehicle_load = len(vehicle.onboard) if vehicle else 0
    capacity = getattr(env.config, 'vehicle_capacity', 10)
    load_ratio = vehicle_load / max(1, capacity)
    
    for i, stop_id in enumerate(candidate_stops):
        if stop_id < 0:  # padding
            continue
        
        # Waiting count
        queue = waiting.get(stop_id, [])
        dynamic[0, i] = len(queue) / 10.0  # Normalize
        
        # Vehicle load (broadcast)
        dynamic[1, i] = load_ratio
        
        # Current time (broadcast)
        dynamic[2, i] = current_time / max_time
    
    return dynamic.reshape(1, 3, kmax)


def _select_candidate_stops(
    features: Dict[str, np.ndarray],
    kmax: int,
) -> Tuple[List[int], np.ndarray]:
    """Select and pad candidate stops to fixed size Kmax.
    
    Strategy S1: Fixed Kmax + Padding
    - First slot: current position (index 0)
    - Next slots: candidate destinations sorted by travel time
    - Pad with -1 for dummy slots
    
    Returns:
        candidate_stops: List of stop IDs (length kmax)
        action_mask: Boolean mask [kmax] (True = valid)
    """
    actions = features["actions"].astype(np.int64)
    edge_features = features["edge_features"]
    mask = features["action_mask"].astype(bool)
    
    # Sort by travel time (ascending)
    if len(edge_features) > 0 and len(edge_features[0]) > 3:
        travel_times = [ef[3] for ef in edge_features]
        sorted_indices = np.argsort(travel_times)
    else:
        sorted_indices = np.arange(len(actions))
    
    # Build candidate list
    candidate_stops = []
    action_mask = np.zeros(kmax, dtype=bool)
    
    for i, idx in enumerate(sorted_indices):
        if i >= kmax:
            break
        candidate_stops.append(int(actions[idx]))
        action_mask[i] = mask[idx]
    
    # Pad if needed
    while len(candidate_stops) < kmax:
        candidate_stops.append(-1)  # Dummy
    
    return candidate_stops, action_mask


# =============================================================================
# Model Loading and Initialization
# =============================================================================

def load_wu2024_model(
    model_path: Optional[str],
    config: Dict[str, Any],
    device: torch.device,
) -> Tuple[Wu2024PointerNet, str]:
    """Load or initialize Wu2024 model.
    
    Supports multiple checkpoint formats:
    1. Unified checkpoint (dict with "model_state_dict" key) - from in-domain training
    2. Direct state_dict - legacy format
    3. None - random initialization
    
    Args:
        model_path: Path to pretrained weights (or None for random init)
        config: Wu2024 configuration dict
        device: Torch device
        
    Returns:
        model: Initialized Wu2024PointerNet
        weights_mode: "trained", "pretrained", or "random_init"
    """
    kmax = int(config.get("kmax", 64))
    static_size = 3  # V0: x, y, travel_time
    dynamic_size = 3  # V0: waiting_count, load, current_time
    hidden_size = int(config.get("hidden_size", 128))
    num_layers = int(config.get("num_layers", 1))
    dropout = float(config.get("dropout", 0.1))
    
    model = Wu2024PointerNet(
        static_size=static_size,
        dynamic_size=dynamic_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        device=device,
    )
    
    weights_mode = str(config.get("weights_mode", "random_init"))
    
    if model_path:
        try:
            checkpoint = torch.load(model_path, map_location=device)
            
            # Check if this is a unified checkpoint or direct state_dict
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # Unified checkpoint format from in-domain training
                model.load_state_dict(checkpoint["model_state_dict"])
                weights_mode = "trained"
            else:
                # Legacy format: direct state_dict
                model.load_state_dict(checkpoint)
                weights_mode = "pretrained"
        except Exception:
            # Fall back to random init
            pass
    
    model.to(device)
    model.eval()
    
    return model, weights_mode


# =============================================================================
# Policy Inference
# =============================================================================

def wu2024_policy(
    env: Any,
    features: Dict[str, np.ndarray],
    model: Wu2024PointerNet,
    config: Dict[str, Any],
    device: torch.device,
    rng: np.random.Generator,
) -> Optional[int]:
    """Wu2024 policy inference for evaluation.
    
    Supports two weights_mode:
    - random_init: Use model forward pass with random weights
    - uniform_logits: Return uniform distribution over valid actions
    
    Args:
        env: EventDrivenEnv instance
        features: Feature dict from env.get_feature_batch()
        model: Wu2024PointerNet model
        config: Wu2024 configuration
        device: Torch device
        rng: Numpy random generator for deterministic sampling
        
    Returns:
        Selected stop ID or None if no valid action
    """
    actions = features["actions"].astype(np.int64)
    mask = features["action_mask"].astype(bool)
    
    if len(actions) == 0:
        return None
    
    valid_indices = np.where(mask)[0]
    if len(valid_indices) == 0:
        return None
    
    kmax = int(config.get("kmax", 64))
    weights_mode = str(config.get("weights_mode", "random_init"))
    
    # Select and pad candidates
    candidate_stops, action_mask = _select_candidate_stops(features, kmax)
    
    # Uniform logits mode: skip model, just sample uniformly
    if weights_mode == "uniform_logits":
        # Select first valid action (deterministic for fixed seed)
        idx = int(valid_indices[0])
        return int(actions[idx])
    
    # Build feature tensors
    static = _build_static_features(env, features, kmax, candidate_stops)
    dynamic = _build_dynamic_features(env, features, kmax, candidate_stops)
    
    # Convert to tensors
    static_t = torch.tensor(static, dtype=torch.float32, device=device)
    dynamic_t = torch.tensor(dynamic, dtype=torch.float32, device=device)
    mask_t = torch.tensor(action_mask.astype(np.float32), dtype=torch.float32, device=device).unsqueeze(0)
    
    # Forward pass
    with torch.no_grad():
        probs = model(static_t, dynamic_t, mask_t)
    
    probs_np = probs.cpu().numpy().flatten()
    
    # Select action: argmax for deterministic evaluation
    # Only consider valid (non-padded) indices
    valid_kmax = [i for i, s in enumerate(candidate_stops) if s >= 0 and action_mask[i]]
    if not valid_kmax:
        return int(actions[valid_indices[0]])
    
    best_idx = max(valid_kmax, key=lambda i: probs_np[i])
    selected_stop = candidate_stops[best_idx]
    
    # Verify selection is in original actions
    if selected_stop in actions:
        return int(selected_stop)
    else:
        # Fallback to first valid action
        return int(actions[valid_indices[0]])
