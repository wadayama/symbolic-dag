"""dag-demo standalone desktop app (pywebview, no HTTP server).

A native desktop window for symbolic-dag. The drawing/rendering layer
(Cytoscape.js + KaTeX) runs in an embedded WebView; the symbolic engine
(:mod:`core`) runs in-process and is reached directly through a JS<->Python
bridge — there is no localhost server. All front-end assets (Cytoscape, KaTeX,
fonts) are vendored under ``static/vendor/``, so the app runs fully offline.

Run:  uv run python desktop.py
"""

from __future__ import annotations

import os
from typing import Any

import webview

from core import ComputeError, GraphIn, run_compute, to_source

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
INDEX_HTML = os.path.join(STATIC_DIR, "index.html")


class Api:
    """Exposed to the page as ``window.pywebview.api``.

    Takes the drawn-graph payload and returns the CMI result. Errors are returned
    as ``{"_error", "_status"}`` so the front-end can render the message.
    """

    def compute(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            return run_compute(GraphIn(**(body or {})))
        except ComputeError as e:
            return {"_error": e.detail, "_status": e.status}
        except Exception as exc:  # validation / unexpected — surface, don't crash
            return {"_error": f"Invalid input: {exc}", "_status": 400}

    def export(self, body: dict[str, Any], level: str = "high") -> dict[str, Any]:
        """Emit runnable symbolic_dag source for the drawn graph (no compute)."""
        try:
            return {"code": to_source(GraphIn(**(body or {})), level)}
        except ComputeError as e:
            return {"_error": e.detail, "_status": e.status}
        except Exception as exc:
            return {"_error": f"Invalid input: {exc}", "_status": 400}


def main() -> None:
    webview.create_window(
        "symbolic-dag demo",
        url=INDEX_HTML,
        js_api=Api(),
        width=1280,
        height=820,
        min_size=(900, 600),
    )
    # http_server=True serves the static/ directory (the entry file's folder) so
    # the page's relative asset URLs (vendor/..., app.js, style.css) resolve.
    webview.start(http_server=True)


if __name__ == "__main__":
    main()
