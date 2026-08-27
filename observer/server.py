from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from observer.trace import TraceRunner


STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


def make_handler(runtime: Any, static_dir: Path | None = None) -> type[BaseHTTPRequestHandler]:
    assets = static_dir or Path(__file__).with_name("static")

    class ObserverHandler(BaseHTTPRequestHandler):
        def _send_json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_static(self, filename: str, content_type: str) -> None:
            body = (assets / filename).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in STATIC_FILES:
                filename, content_type = STATIC_FILES[parsed.path]
                self._send_static(filename, content_type)
                return
            if parsed.path == "/api/health":
                self._send_json({"status": "ok"})
                return
            if parsed.path == "/api/sessions":
                self._send_json(runtime.list_sessions())
                return
            if parsed.path == "/api/trace":
                query = parse_qs(parsed.query)
                sample_id = (query.get("sample_id") or [""])[0]
                if not sample_id:
                    self._send_json({"error": "sample_id is required"}, 400)
                    return
                try:
                    payload = runtime.trace(sample_id, refresh=(query.get("refresh") == ["1"]))
                except KeyError as exc:
                    self._send_json({"error": str(exc)}, 404)
                    return
                except Exception as exc:
                    self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)
                    return
                self._send_json(payload)
                return
            self._send_json({"error": "not found"}, 404)

        def log_message(self, format: str, *args: object) -> None:
            print(f"[observer] {self.address_string()} - {format % args}")

    return ObserverHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Shopping Copilot layer observer")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--results", default="results.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    print("Loading catalog, public sessions, Agent index, and prior results...")
    runtime = TraceRunner.from_paths(args.catalog, args.dataset, args.results)
    server = HTTPServer((args.host, args.port), make_handler(runtime))
    url = f"http://{args.host}:{args.port}"
    print(f"IntentGraph Layer Observer: {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping observer.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
