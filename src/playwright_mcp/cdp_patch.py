"""
Minimal, mac-safe CDP transport patch.

Goal: reduce visible control-channel fingerprints (e.g., Playwright
headers) without breaking protocol. This is intentionally conservative:
- strips obvious "User-Agent" header from the websocket handshake
- leaves Sec-WebSocket-Protocol intact to avoid breaking Chromium
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _patch_websocket_transport() -> None:
    try:
        from playwright._impl._transport import WebSocketTransport
    except Exception as exc:  # pragma: no cover
        logger.debug("CDP patch skipped: cannot import WebSocketTransport: %s", exc)
        return

    original_init = WebSocketTransport.__init__

    def patched_init(self, *, ws_endpoint: str, headers: Dict[str, str] | None = None, **kwargs: Any):
        headers = dict(headers or {})
        headers.pop("User-Agent", None)  # drop Playwright UA marker in handshake
        return original_init(self, ws_endpoint=ws_endpoint, headers=headers, **kwargs)

    WebSocketTransport.__init__ = patched_init  # type: ignore[assignment]


def apply() -> None:
    """Apply CDP transport tweaks. Safe to call multiple times."""
    try:
        _patch_websocket_transport()
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to apply CDP transport patch: %s", exc)

