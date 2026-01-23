"""Realtime visualization dashboard for EdgeQ training."""

from __future__ import annotations

import argparse
import json
import threading
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import zmq
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pyzmq is required for realtime viz.") from exc

try:
    import dash
    from dash import dcc, html
except ImportError as exc:  # pragma: no cover
    raise SystemExit("dash is required for realtime viz.") from exc

import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config
from src.utils.build_info import get_build_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--zmq-url", default="tcp://127.0.0.1:5555")
    parser.add_argument("--topic", default="bus")
    parser.add_argument("--history-len", type=int, default=500)
    return parser.parse_args()


REWARD_COMPONENT_KEYS = [
    "reward_base_service",
    "reward_waiting_churn_penalty",
    "reward_fairness_penalty",
    "reward_cvar_penalty",
    "reward_travel_cost",
    "reward_onboard_delay_penalty",
    "reward_onboard_churn_penalty",
    "reward_backlog_penalty",
    "reward_waiting_time_penalty",
    "reward_potential_shaping",
    "reward_potential_shaping_raw",
    "reward_congestion_penalty",
    "reward_tacc_bonus",
]


class StreamState:
    def __init__(self, history_len: int) -> None:
        self.history_len = int(history_len)
        self.lock = threading.Lock()
        self.latest: Optional[Dict[str, Any]] = None
        self.reward_history = deque(maxlen=2000)
        self.entropy_history = deque(maxlen=2000)
        self.step_history = deque(maxlen=2000)
        self.event_flags = deque(maxlen=2000)
        self.waiting_churn_prob_mean = deque(maxlen=2000)
        self.onboard_churn_prob_mean = deque(maxlen=2000)
        self.reward_components = {key: deque(maxlen=2000) for key in REWARD_COMPONENT_KEYS}

    def update(self, payload: Dict[str, Any]) -> None:
        with self.lock:
            self.latest = payload
            step = int(payload.get("global_step", payload.get("step", 0)))
            reward_total = float(payload.get("reward_terms", {}).get("reward_total", 0.0))
            entropy = float(payload.get("q_entropy", 0.0))
            self.step_history.append(step)
            self.reward_history.append(reward_total)
            self.entropy_history.append(entropy)

            reward_terms = payload.get("reward_terms", {})
            for key in REWARD_COMPONENT_KEYS:
                self.reward_components[key].append(float(reward_terms.get(key, 0.0)))

            event_count = (
                float(payload.get("step_served", 0.0))
                + float(payload.get("step_waiting_churned", 0.0))
                + float(payload.get("step_onboard_churned", 0.0))
                + float(payload.get("step_waiting_timeouts", 0.0))
            )
            self.event_flags.append(1.0 if event_count > 0.0 else 0.0)
            self.waiting_churn_prob_mean.append(float(payload.get("step_waiting_churn_prob_mean", 0.0)))
            self.onboard_churn_prob_mean.append(float(payload.get("step_onboard_churn_prob_mean", 0.0)))


def _start_receiver(state: StreamState, zmq_url: str, topic: str) -> threading.Thread:
    context = zmq.Context.instance()
    socket = context.socket(zmq.SUB)
    socket.connect(zmq_url)
    socket.setsockopt_string(zmq.SUBSCRIBE, topic)
    socket.RCVTIMEO = 500

    def _loop() -> None:
        while True:
            try:
                parts = socket.recv_multipart()
            except zmq.Again:
                continue
            if len(parts) != 2:
                continue
            try:
                payload = json.loads(parts[1].decode("utf-8"))
            except Exception:
                continue
            state.update(payload)

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return thread




def _make_series_figure(xs: list, ys: list, title: str, color: str) -> go.Figure:
    fig = go.Figure()
    if xs and ys:
        fig.add_trace(
            go.Scattergl(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=color, width=2),
            )
        )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        title=title,
        xaxis=dict(title="global_step"),
        yaxis=dict(title=title),
        plot_bgcolor="#f4f3ef",
        paper_bgcolor="#f4f3ef",
    )
    return fig


def _rolling_mean(values: list, window: int) -> list:
    if not values:
        return []
    window = max(1, int(window))
    out = []
    acc = 0.0
    for idx, val in enumerate(values):
        acc += float(val)
        if idx >= window:
            acc -= float(values[idx - window])
        denom = float(min(window, idx + 1))
        out.append(acc / denom if denom > 0 else 0.0)
    return out


def _make_components_figure(steps: list, components: Dict[str, list]) -> go.Figure:
    fig = go.Figure()
    palette = [
        "#1f6f8b",
        "#8c2f39",
        "#5a7f37",
        "#d65f5f",
        "#7a8c6d",
        "#556b2f",
        "#b85c38",
        "#2a9d8f",
        "#e9c46a",
        "#264653",
        "#9b5de5",
    ]
    for idx, key in enumerate(REWARD_COMPONENT_KEYS):
        ys = components.get(key, [])
        if not ys:
            continue
        fig.add_trace(
            go.Scattergl(
                x=steps,
                y=ys,
                mode="lines",
                line=dict(color=palette[idx % len(palette)], width=1.5),
                name=key,
            )
        )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        title="reward_components",
        xaxis=dict(title="global_step"),
        yaxis=dict(title="value"),
        plot_bgcolor="#f4f3ef",
        paper_bgcolor="#f4f3ef",
        legend=dict(orientation="h"),
    )
    return fig


def _make_event_rate_figure(steps: list, event_flags: list, window: int) -> go.Figure:
    rates = _rolling_mean(event_flags, window)
    fig = go.Figure()
    if steps and rates:
        fig.add_trace(
            go.Scattergl(
                x=steps,
                y=rates,
                mode="lines",
                line=dict(color="#2a9d8f", width=2),
            )
        )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        title=f"effective_event_rate (window={window})",
        xaxis=dict(title="global_step"),
        yaxis=dict(title="rate"),
        plot_bgcolor="#f4f3ef",
        paper_bgcolor="#f4f3ef",
        yaxis_range=[0.0, 1.0],
    )
    return fig


def _make_churn_prob_figure(steps: list, waiting_means: list, onboard_means: list) -> go.Figure:
    fig = go.Figure()
    if steps and waiting_means:
        fig.add_trace(
            go.Scattergl(
                x=steps,
                y=waiting_means,
                mode="lines",
                line=dict(color="#1f6f8b", width=2),
                name="waiting_churn_prob_mean",
            )
        )
    if steps and onboard_means:
        fig.add_trace(
            go.Scattergl(
                x=steps,
                y=onboard_means,
                mode="lines",
                line=dict(color="#d65f5f", width=2),
                name="onboard_churn_prob_mean",
            )
        )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        title="churn_prob_mean",
        xaxis=dict(title="global_step"),
        yaxis=dict(title="probability"),
        plot_bgcolor="#f4f3ef",
        paper_bgcolor="#f4f3ef",
        legend=dict(orientation="h"),
    )
    return fig


def _format_value(latest: Dict[str, Any], key: str, fmt: str) -> str:
    if key not in latest:
        return "MISSING"
    value = latest.get(key)
    try:
        return fmt.format(value)
    except Exception:
        return "MISSING"


def _format_reward_term(latest: Dict[str, Any], key: str, fmt: str) -> str:
    reward_terms = latest.get("reward_terms")
    if not isinstance(reward_terms, dict) or key not in reward_terms:
        return "MISSING"
    try:
        return fmt.format(reward_terms.get(key))
    except Exception:
        return "MISSING"


def _format_metrics(latest: Optional[Dict[str, Any]]) -> str:
    if not latest:
        return "Waiting for data..."
    alerts = latest.get("alerts", [])
    alert_lines = []
    if alerts:
        for alert in alerts:
            code = alert.get("code", "unknown")
            severity = alert.get("severity", "n/a")
            message = alert.get("message", "")
            dump_path = alert.get("dump_path")
            suffix = f" (dump={dump_path})" if dump_path else ""
            alert_lines.append(f"- [{severity}] {code}: {message}{suffix}")
    alert_block = "\n".join(alert_lines) if alert_lines else "none"
    lines = [
        f"build_id: {_format_value(latest, 'build_id', '{}')}",
        f"global_step: {_format_value(latest, 'global_step', '{:d}')}",
        f"episode: {_format_value(latest, 'episode_index', '{:d}')} / steps: {_format_value(latest, 'episode_steps', '{:d}')}",
        f"sim_time_sec: {_format_value(latest, 'current_time', '{:.1f}')}",
        f"active_vehicle_id: {_format_value(latest, 'active_vehicle_id', '{}')}",
        f"ready_vehicles: {_format_value(latest, 'ready_vehicles', '{:d}')}",
        f"event_queue_len: {_format_value(latest, 'event_queue_len', '{:d}')}",
        f"env_steps: {_format_value(latest, 'env_steps', '{:d}')}",
        f"epsilon: {_format_value(latest, 'epsilon', '{:.3f}')}",
        f"reward_total: {_format_reward_term(latest, 'reward_total', '{:.3f}')}",
        f"served: {_format_value(latest, 'served', '{:.1f}')}",
        f"waiting_churned: {_format_value(latest, 'waiting_churned', '{:.1f}')}",
        f"onboard_churned: {_format_value(latest, 'onboard_churned', '{:.1f}')}",
        f"waiting_remaining: {_format_value(latest, 'waiting_remaining', '{:.1f}')}",
        f"onboard_remaining: {_format_value(latest, 'onboard_remaining', '{:.1f}')}",
        f"q_entropy: {_format_value(latest, 'q_entropy', '{:.3f}')}",
        f"q_entropy_norm: {_format_value(latest, 'q_entropy_norm', '{:.3f}')}",
        f"epsilon_entropy: {_format_value(latest, 'epsilon_entropy', '{:.3f}')}",
        f"waiting_churn_prob_mean: {_format_value(latest, 'step_waiting_churn_prob_mean', '{:.3e}')}",
        f"onboard_churn_prob_mean: {_format_value(latest, 'step_onboard_churn_prob_mean', '{:.3e}')}",
        f"reward_potential_alpha: {_format_value(latest, 'reward_potential_alpha', '{:.6f}')}",
        f"reward_potential_alpha_source: {_format_value(latest, 'reward_potential_alpha_source', '{}')}",
        f"reward_potential_lost_weight: {_format_value(latest, 'reward_potential_lost_weight', '{:.6f}')}",
        f"reward_potential_scale_with_reward_scale: {_format_value(latest, 'reward_potential_scale_with_reward_scale', '{}')}",
        f"phi_before: {_format_value(latest, 'phi_before', '{:.6e}')}",
        f"phi_after: {_format_value(latest, 'phi_after', '{:.6e}')}",
        f"phi_delta: {_format_value(latest, 'phi_delta', '{:.6e}')}",
        f"phi_backlog_before: {_format_value(latest, 'phi_backlog_before', '{:.6e}')}",
        f"phi_backlog_after: {_format_value(latest, 'phi_backlog_after', '{:.6e}')}",
        f"lost_total_before: {_format_value(latest, 'lost_total_before', '{:.6e}')}",
        f"lost_total_after: {_format_value(latest, 'lost_total_after', '{:.6e}')}",
        f"waiting_churned_before: {_format_value(latest, 'waiting_churned_before', '{:.6e}')}",
        f"waiting_churned_after: {_format_value(latest, 'waiting_churned_after', '{:.6e}')}",
        f"onboard_churned_before: {_format_value(latest, 'onboard_churned_before', '{:.6e}')}",
        f"onboard_churned_after: {_format_value(latest, 'onboard_churned_after', '{:.6e}')}",
        f"structural_before: {_format_value(latest, 'structural_unserviceable_before', '{:.6e}')}",
        f"structural_after: {_format_value(latest, 'structural_unserviceable_after', '{:.6e}')}",
        f"waiting_before: {_format_value(latest, 'waiting_remaining_before', '{:.6e}')}",
        f"waiting_after: {_format_value(latest, 'waiting_remaining_after', '{:.6e}')}",
        f"onboard_before: {_format_value(latest, 'onboard_remaining_before', '{:.6e}')}",
        f"onboard_after: {_format_value(latest, 'onboard_remaining_after', '{:.6e}')}",
        f"reward_potential_shaping_raw: {_format_reward_term(latest, 'reward_potential_shaping_raw', '{:.6e}')}",
        f"reward_potential_shaping: {_format_reward_term(latest, 'reward_potential_shaping', '{:.6e}')}",
        f"action_count: {_format_value(latest, 'action_count', '{:d}')}",
        f"action_valid_ratio: {_format_value(latest, 'action_valid_ratio', '{:.3f}')}",
        f"stop_ratio: {_format_value(latest, 'stop_ratio', '{:.3f}')}",
        f"served_per_decision: {_format_value(latest, 'served_per_decision', '{:.3f}')}",
        f"waiting_churn_per_decision: {_format_value(latest, 'waiting_churn_per_decision', '{:.3f}')}",
        f"invalid_action_ratio: {_format_value(latest, 'invalid_action_ratio', '{:.3f}')}",
        f"reward_nonzero_ratio: {_format_value(latest, 'reward_nonzero_ratio', '{:.3f}')}",
        f"action_stop: {_format_value(latest, 'action_stop', '{}')}",
        f"done: {_format_value(latest, 'done', '{}')} ({_format_value(latest, 'done_reason', '{}')})",
        "alerts:",
        alert_block,
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    load_config(args.config)
    build_id = get_build_id()
    print(f"BUILD_ID={build_id}")
    state = StreamState(history_len=args.history_len)
    _start_receiver(state, args.zmq_url, args.topic)

    app = dash.Dash(__name__)
    app.layout = html.Div(
        style={
            "backgroundColor": "#f4f3ef",
            "padding": "12px",
            "fontFamily": "Verdana, Geneva, sans-serif",
            "color": "#2b2b2b",
        },
        children=[
            html.Div(
                style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
                children=[
                    html.Div(
                        style={"flex": "1 1 360px", "minWidth": "320px"},
                        children=[
                            html.H2("Realtime Bus Training View"),
                            html.Div(f"BUILD_ID: {build_id}", style={"fontWeight": "bold"}),
                            html.Pre(id="metrics", style={"whiteSpace": "pre-wrap"}),
                        ],
                    ),
                    html.Div(
                        style={"flex": "2 1 640px", "minWidth": "320px"},
                        children=[dcc.Graph(id="reward_components")],
                    ),
                ],
            ),
            html.Div(
                style={"display": "flex", "gap": "12px", "marginTop": "12px", "flexWrap": "wrap"},
                children=[
                    html.Div(style={"flex": "1 1 420px"}, children=[dcc.Graph(id="reward_total")]),
                    html.Div(style={"flex": "1 1 420px"}, children=[dcc.Graph(id="entropy")]),
                    html.Div(style={"flex": "1 1 420px"}, children=[dcc.Graph(id="event_rate")]),
                    html.Div(style={"flex": "1 1 420px"}, children=[dcc.Graph(id="churn_prob")]),
                ],
            ),
            dcc.Interval(id="tick", interval=1000, n_intervals=0),
        ],
    )

    @app.callback(
        [
            dash.Output("metrics", "children"),
            dash.Output("reward_components", "figure"),
            dash.Output("reward_total", "figure"),
            dash.Output("entropy", "figure"),
            dash.Output("event_rate", "figure"),
            dash.Output("churn_prob", "figure"),
        ],
        [dash.Input("tick", "n_intervals")],
    )
    def _update(_n: int):
        with state.lock:
            latest = dict(state.latest) if state.latest else None
            steps = list(state.step_history)
            rewards = list(state.reward_history)
            entropies = list(state.entropy_history)
            event_flags = list(state.event_flags)
            waiting_churn_means = list(state.waiting_churn_prob_mean)
            onboard_churn_means = list(state.onboard_churn_prob_mean)
            reward_components = {key: list(value) for key, value in state.reward_components.items()}
        metrics = _format_metrics(latest)
        fig_reward_components = _make_components_figure(steps, reward_components)
        fig_reward = _make_series_figure(steps, rewards, "reward_total", "#1f6f8b")
        fig_entropy = _make_series_figure(steps, entropies, "q_entropy", "#d65f5f")
        fig_event_rate = _make_event_rate_figure(steps, event_flags, window=200)
        fig_churn_prob = _make_churn_prob_figure(steps, waiting_churn_means, onboard_churn_means)
        return metrics, fig_reward_components, fig_reward, fig_entropy, fig_event_rate, fig_churn_prob

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
