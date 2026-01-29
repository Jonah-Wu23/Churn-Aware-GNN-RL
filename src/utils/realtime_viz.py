"""Realtime visualization publisher for training runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import time
from typing import Any, Dict, Optional

try:
    import zmq
except ImportError:  # pragma: no cover - optional dependency
    zmq = None

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class RealtimeVizConfig:
    enabled: bool = False
    zmq_url: str = "tcp://127.0.0.1:5555"
    topic: str = "bus"
    publish_every_steps: int = 5
    publish_on_episode_end: bool = True
    bind: bool = True

    @staticmethod
    def from_dict(raw: Optional[Dict[str, Any]]) -> "RealtimeVizConfig":
        if not raw:
            return RealtimeVizConfig()
        return RealtimeVizConfig(
            enabled=bool(raw.get("enabled", False)),
            zmq_url=str(raw.get("zmq_url", "tcp://127.0.0.1:5555")),
            topic=str(raw.get("topic", "bus")),
            publish_every_steps=int(raw.get("publish_every_steps", 5)),
            publish_on_episode_end=bool(raw.get("publish_on_episode_end", True)),
            bind=bool(raw.get("bind", True)),
        )


class RealtimeVizPublisher:
    def __init__(self, config: RealtimeVizConfig) -> None:
        self.config = config
        self.enabled = bool(config.enabled)
        self._context = None
        self._socket = None
        if not self.enabled:
            return
        if zmq is None:
            LOG.warning("pyzmq not installed; realtime viz disabled.")
            self.enabled = False
            return
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.linger = 0
        try:
            if self.config.bind:
                self._socket.bind(self.config.zmq_url)
            else:
                self._socket.connect(self.config.zmq_url)
        except Exception as exc:  # pragma: no cover - depends on runtime
            LOG.warning("Realtime viz socket init failed: %s", exc)
            self.enabled = False
            try:
                self._socket.close(0)
            except Exception:
                pass
            self._socket = None

    def publish(self, payload: Dict[str, Any]) -> None:
        if not self.enabled or self._socket is None or zmq is None:
            return
        try:
            message = {
                "ts": time.time(),
                **payload,
            }
            data = json.dumps(message, ensure_ascii=True).encode("utf-8")
            self._socket.send_multipart(
                [self.config.topic.encode("utf-8"), data],
                flags=zmq.DONTWAIT,
            )
        except Exception:
            # Drop on any errors to avoid impacting training loop.
            return

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close(0)
            except Exception:
                pass
        self._socket = None
