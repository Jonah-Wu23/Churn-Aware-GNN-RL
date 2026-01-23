"""Realtime visualization dashboard for EdgeQ training."""

from __future__ import annotations

import argparse
import json
import threading
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, List

try:
    import zmq
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pyzmq is required for realtime viz.") from exc

try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output
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
        self.action_valid_count_history = deque(maxlen=2000)
        self.action_valid_ratio_history = deque(maxlen=2000)

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
            self.action_valid_count_history.append(int(payload.get("action_mask_valid_count", 0)))
            self.action_valid_ratio_history.append(float(payload.get("action_valid_ratio", 0.0)))


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


# -----------------------------------------------------------------------------
# Formatting & Computation Helpers
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Modern Visuals (Figures)
# -----------------------------------------------------------------------------

COLOR_AXIS = "#94a3b8"  # slate-400
COLOR_GRID = "#334155"  # slate-700 (faint)
FONT_FAMILY = "Inter, system-ui, sans-serif"
BG_PAPER = "rgba(0,0,0,0)"
BG_PLOT = "rgba(0,0,0,0)"

def _apply_dark_theme(fig: go.Figure, title: str, y_title: str = None) -> None:
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=40, r=20, t=40, b=40),
        title=dict(text=title, font=dict(size=14, color="#e2e8f0")),
        xaxis=dict(
            title="Global Step", 
            showgrid=False, 
            color=COLOR_AXIS
        ),
        yaxis=dict(
            title=y_title if y_title else title, 
            gridcolor=COLOR_GRID, 
            color=COLOR_AXIS,
            showgrid=True
        ),
        plot_bgcolor=BG_PLOT,
        paper_bgcolor=BG_PAPER,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)"
        )
    )

def _make_series_figure(xs: list, ys: list, title: str, color: str) -> go.Figure:
    fig = go.Figure()
    if xs and ys:
        fig.add_trace(
            go.Scattergl(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=color, width=2),
                fill='tozeroy',  # Area chart effect
                fillcolor=color.replace(")", ", 0.1)").replace("rgb", "rgba") if "rgb" in color else color,
            )
        )
    _apply_dark_theme(fig, title)
    return fig


def _make_components_figure(steps: list, components: Dict[str, list]) -> go.Figure:
    fig = go.Figure()
    # Cyberpunk palette
    palette = [
        "#22d3ee", # cyan-400
        "#f472b6", # pink-400
        "#a78bfa", # violet-400
        "#34d399", # emerald-400
        "#fbbf24", # amber-400
        "#f87171", # red-400
        "#60a5fa", # blue-400
        "#c084fc", # purple-400
        "#e879f9", # fuchsia-400
        "#818cf8", # indigo-400
    ]
    
    for idx, key in enumerate(REWARD_COMPONENT_KEYS):
        ys = components.get(key, [])
        if not ys:
            continue
        # Only show last non-zero values effectively? No, show all history
        fig.add_trace(
            go.Scattergl(
                x=steps,
                y=ys,
                mode="lines",
                line=dict(color=palette[idx % len(palette)], width=1.5),
                name=key.replace("reward_", ""),
            )
        )
    _apply_dark_theme(fig, "Reward Components", "Value")
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
                line=dict(color="#2dd4bf", width=2), # teal-400
                name="Rate",
                fill='tozeroy',
                fillcolor="rgba(45, 212, 191, 0.1)"
            )
        )
    _apply_dark_theme(fig, f"Effective Event Rate (w={window})", "Rate")
    fig.update_yaxes(range=[0.0, 1.0])
    return fig


def _make_churn_prob_figure(steps: list, waiting_means: list, onboard_means: list) -> go.Figure:
    fig = go.Figure()
    if steps and waiting_means:
        fig.add_trace(
            go.Scattergl(
                x=steps,
                y=waiting_means,
                mode="lines",
                line=dict(color="#38bdf8", width=2), # sky-400
                name="Waiting Churn",
            )
        )
    if steps and onboard_means:
        fig.add_trace(
            go.Scattergl(
                x=steps,
                y=onboard_means,
                mode="lines",
                line=dict(color="#f43f5e", width=2), # rose-500
                name="Onboard Churn",
            )
        )
    _apply_dark_theme(fig, "Churn Probabilities", "Prob")
    return fig

# -----------------------------------------------------------------------------
# UI Components
# -----------------------------------------------------------------------------

STATS_CARD_STYLE = {
    "backgroundColor": "#1e293b", # slate-800
    "borderRadius": "8px",
    "padding": "16px",
    "border": "1px solid #334155", # slate-700
    "boxShadow": "0 4px 6px -1px rgba(0, 0, 0, 0.3)",
    "display": "flex",
    "flexDirection": "column",
    "minWidth": "140px",
    "flex": "1"
}

STATS_LABEL_STYLE = {
    "color": "#94a3b8", # slate-400
    "fontSize": "12px",
    "fontWeight": "600",
    "textTransform": "uppercase",
    "letterSpacing": "0.05em",
    "marginBottom": "4px"
}

STATS_VALUE_STYLE = {
    "color": "#f1f5f9", # slate-100
    "fontSize": "24px",
    "fontWeight": "700",
    "fontFamily": "Monaco, Consolas, monospace"
}

STATS_SUB_STYLE = {
    "color": "#64748b", # slate-500
    "fontSize": "12px",
    "marginTop": "4px"
}

def _make_stat_card(label: str, value: str, subtext: str = None, color: str = None) -> html.Div:
    val_style = STATS_VALUE_STYLE.copy()
    if color:
        val_style["color"] = color
    
    children = [
        html.Div(label, style=STATS_LABEL_STYLE),
        html.Div(value, style=val_style),
    ]
    if subtext:
        children.append(html.Div(subtext, style=STATS_SUB_STYLE))
        
    return html.Div(children, style=STATS_CARD_STYLE)

def _format_val(d: Optional[Dict], k: str, f: str = "{}") -> str:
    if not d or k not in d: return "-"
    try: return f.format(d[k])
    except: return str(d[k])

# -----------------------------------------------------------------------------
# Main Visual Update Logic
# -----------------------------------------------------------------------------

def _generate_layout(build_id: str) -> html.Div:
    return html.Div(
        style={
            "backgroundColor": "#0f172a", # slate-900
            "minHeight": "100vh",
            "padding": "24px",
            "fontFamily": "Inter, system-ui, sans-serif",
            "color": "#e2e8f0",
        },
        children=[
            # Header
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "24px"},
                children=[
                    html.Div([
                        html.H1("EdgeQ Training Monitor", style={"margin": 0, "fontSize": "24px", "fontWeight": "800", "background": "linear-gradient(to right, #38bdf8, #818cf8)", "-webkit-background-clip": "text", "-webkit-text-fill-color": "transparent"}),
                        html.Div(f"Build: {build_id}", style={"color": "#64748b", "fontSize": "12px", "marginTop": "4px"}),
                    ]),
                    html.Div(id="clock-display", style={"fontSize": "14px", "color": "#94a3b8"})
                ]
            ),
            
            # Top Stats Row
            html.Div(
                id="stats-row-1",
                style={"display": "flex", "gap": "12px", "marginBottom": "12px", "flexWrap": "wrap"},
                children="waiting for data..." 
            ),
             # Second Stats Row
            html.Div(
                id="stats-row-2",
                style={"display": "flex", "gap": "12px", "marginBottom": "24px", "flexWrap": "wrap"},
                children="..." 
            ),

            # Main Grid
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(450px, 1fr))", "gap": "24px"},
                children=[
                    # Large Reward Components Plot
                    html.Div(
                        style={"gridColumn": "1 / -1", "backgroundColor": "#1e293b", "borderRadius": "12px", "padding": "16px", "border": "1px solid #334155"},
                        children=[dcc.Graph(id="reward_components", style={"height": "400px"})]
                    ),
                    
                    # 4 Quadrant Plots
                    html.Div(style={"backgroundColor": "#1e293b", "borderRadius": "12px", "padding": "12px", "border": "1px solid #334155"},
                             children=[dcc.Graph(id="reward_total", style={"height": "300px"})]),
                    html.Div(style={"backgroundColor": "#1e293b", "borderRadius": "12px", "padding": "12px", "border": "1px solid #334155"},
                             children=[dcc.Graph(id="entropy", style={"height": "300px"})]),
                    html.Div(style={"backgroundColor": "#1e293b", "borderRadius": "12px", "padding": "12px", "border": "1px solid #334155"},
                             children=[dcc.Graph(id="event_rate", style={"height": "300px"})]),
                    html.Div(style={"backgroundColor": "#1e293b", "borderRadius": "12px", "padding": "12px", "border": "1px solid #334155"},
                             children=[dcc.Graph(id="churn_prob", style={"height": "300px"})]),
                             
                    # Additional Analysis Plots
                    html.Div(style={"backgroundColor": "#1e293b", "borderRadius": "12px", "padding": "12px", "border": "1px solid #334155"},
                             children=[dcc.Graph(id="action_valid_count", style={"height": "300px"})]),
                    html.Div(style={"backgroundColor": "#1e293b", "borderRadius": "12px", "padding": "12px", "border": "1px solid #334155"},
                             children=[dcc.Graph(id="action_valid_ratio", style={"height": "300px"})]),
                ]
            ),
            
            # Alert Log
            html.Div(
                style={"marginTop": "24px", "backgroundColor": "#1e293b", "borderRadius": "12px", "border": "1px solid #334155", "overflow": "hidden"},
                children=[
                    html.Div("System Alerts / Logs", style={"padding": "12px 16px", "borderBottom": "1px solid #334155", "fontWeight": "600", "color": "#94a3b8"}),
                    html.Div(id="alert-log", style={"padding": "16px", "maxHeight": "200px", "overflowY": "auto", "fontFamily": "monospace", "fontSize": "13px"})
                ]
            ),

            dcc.Interval(id="tick", interval=1000, n_intervals=0),
        ]
    )

def main() -> None:
    args = parse_args()
    load_config(args.config)
    build_id = get_build_id()
    print(f"BUILD_ID={build_id}")
    print(f"Listening on {args.zmq_url} for topic '{args.topic}'")
    
    state = StreamState(history_len=args.history_len)
    _start_receiver(state, args.zmq_url, args.topic)

    app = dash.Dash(__name__, title="EdgeQ Monitor")
    app.layout = _generate_layout(build_id)

    @app.callback(
        [
            Output("stats-row-1", "children"),
            Output("stats-row-2", "children"),
            Output("clock-display", "children"),
            Output("alert-log", "children"),
            Output("reward_components", "figure"),
            Output("reward_total", "figure"),
            Output("entropy", "figure"),
            Output("event_rate", "figure"),
            Output("churn_prob", "figure"),
            Output("action_valid_count", "figure"),
            Output("action_valid_ratio", "figure"),
        ],
        [Input("tick", "n_intervals")],
    )
    def _update(_n: int):
        with state.lock:
            latest = dict(state.latest) if state.latest else None
            steps = list(state.step_history)
            
            # Snapshots for plotting
            rewards = list(state.reward_history)
            entropies = list(state.entropy_history)
            event_flags = list(state.event_flags)
            waiting_churn_means = list(state.waiting_churn_prob_mean)
            onboard_churn_means = list(state.onboard_churn_prob_mean)
            reward_components = {key: list(value) for key, value in state.reward_components.items()}
            action_valid_counts = list(state.action_valid_count_history)
            action_valid_ratios = list(state.action_valid_ratio_history)

        # 1. Stats Rows
        if not latest:
            return ([], [], "Waiting...", "No data received yet.", go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure())
        
        reward_terms = latest.get("reward_terms")
        if not isinstance(reward_terms, dict):
            reward_terms = {}
        r_val = reward_terms.get("reward_total", 0.0)

        # Row 1: High Level
        row1 = [
            _make_stat_card("Episode", f"{_format_val(latest, 'episode_index')} / {_format_val(latest, 'episode_steps')}", "Steps", "#60a5fa"),
            _make_stat_card("Global Step", _format_val(latest, 'global_step', '{:,}'), color="#c084fc"),
            _make_stat_card("Epsilon", _format_val(latest, 'epsilon', '{:.4f}'), "Exploration", "#fbbf24"),
            _make_stat_card("Entropy", _format_val(latest, 'q_entropy', '{:.3f}'), f"Norm: {_format_val(latest, 'q_entropy_norm', '{:.2f}')}", "#f472b6"),
            _make_stat_card("Reward Total", f"{float(r_val):.3f}", "Current Step", "#34d399"),
        ]

        # Row 2: Operational
        row2 = [
            _make_stat_card("Served", _format_val(latest, 'served', '{:.1f}'), "Pax", "#22d3ee"),
            _make_stat_card("Waiting (Rem)", _format_val(latest, 'waiting_remaining', '{:.1f}'), f"Churn: {_format_val(latest, 'waiting_churned', '{:.1f}')}", "#f87171"),
            _make_stat_card("Onboard (Rem)", _format_val(latest, 'onboard_remaining', '{:.1f}'), f"Churn: {_format_val(latest, 'onboard_churned', '{:.1f}')}", "#f87171"),
            _make_stat_card("Active Vehicle", _format_val(latest, 'active_vehicle_id'), None, "#94a3b8"),
            _make_stat_card("Action Ratio", _format_val(latest, 'action_valid_ratio', '{:.1%}'), f"Count: {_format_val(latest, 'action_mask_valid_count')}", "#a78bfa"),
        ]

        # 2. Clock
        clock = f"Simon Time: {_format_val(latest, 'current_time', '{:.1f}')}s | Env Steps: {_format_val(latest, 'env_steps')}"

        # 3. Alerts
        alerts = latest.get("alerts", [])
        alert_children = []
        if alerts:
            for alert in alerts:
                sev = alert.get("severity", "info").upper()
                col = "#ef4444" if sev == "ERROR" else "#eab308" if sev == "WARNING" else "#3b82f6"
                alert_children.append(html.Div([
                    html.Span(f"[{sev}] {alert.get('code')}: ", style={"color": col, "fontWeight": "bold"}),
                    html.Span(alert.get('message'), style={"color": "#cbd5e1"})
                ], style={"marginBottom": "4px"}))
        else:
            alert_children = html.Div("No active system alerts.", style={"color": "#475569", "fontStyle": "italic"})

        # 4. Plots
        fig_rc = _make_components_figure(steps, reward_components)
        fig_rt = _make_series_figure(steps, rewards, "Total Reward", "#34d399")
        fig_ent = _make_series_figure(steps, entropies, "Entropy", "#f472b6")
        fig_er = _make_event_rate_figure(steps, event_flags, window=200)
        fig_cp = _make_churn_prob_figure(steps, waiting_churn_means, onboard_churn_means)
        fig_ac = _make_series_figure(steps, action_valid_counts, "Valid Action Count", "#fb923c") # orange
        fig_ar = _make_series_figure(steps, action_valid_ratios, "Valid Action Ratio", "#2dd4bf") # teal

        return row1, row2, clock, alert_children, fig_rc, fig_rt, fig_ent, fig_er, fig_cp, fig_ac, fig_ar

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
