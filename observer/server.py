from __future__ import annotations

import argparse
import json
import secrets
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from observer.runtime import StaleRuntimeError, WorkbenchRuntime


STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


class ExclusiveHTTPServer(HTTPServer):
    # SO_EXCLUSIVEADDRUSE and SO_REUSEADDR are mutually exclusive on Windows.
    # HTTPServer enables address reuse by default, so disable that inherited
    # behavior before asking Windows for an exclusive loopback listener.
    allow_reuse_address = False

    def server_bind(self) -> None:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def make_handler(
    runtime: Any,
    static_dir: Path | None = None,
    api_token: str | None = None,
) -> type[BaseHTTPRequestHandler]:
    assets = static_dir or Path(__file__).with_name("static")
    control_token = api_token or secrets.token_urlsafe(32)

    class ObserverHandler(BaseHTTPRequestHandler):
        def _request_is_local(self) -> bool:
            loopback_names = {"127.0.0.1", "localhost", "::1"}
            try:
                host = urlparse(f"//{self.headers.get('Host', '')}")
                if host.hostname not in loopback_names:
                    return False
                if host.port is not None and host.port != self.server.server_port:
                    return False
                origin_value = self.headers.get("Origin")
                if origin_value:
                    origin = urlparse(origin_value)
                    if origin.scheme != "http" or origin.hostname not in loopback_names:
                        return False
                    if origin.port is not None and origin.port != self.server.server_port:
                        return False
            except ValueError:
                return False
            return self.headers.get("Sec-Fetch-Site", "").lower() != "cross-site"

        def _guard_api_request(self, path: str) -> bool:
            if path.startswith("/api/") and not self._request_is_local():
                self._send_json({"error": "cross-site or non-loopback request rejected"}, 403)
                return False
            if (
                path.startswith("/api/")
                and path not in {"/api/health", "/api/token"}
                and not secrets.compare_digest(
                    self.headers.get("X-Observer-Token", ""), control_token
                )
            ):
                self._send_json({"error": "missing or invalid local control token"}, 403)
                return False
            return True

        def _send_json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def _send_static(self, filename: str, content_type: str) -> None:
            body = (assets / filename).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                raise ValueError("Content-Type must be application/json")
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0:
                raise ValueError("Content-Length must not be negative")
            if length > 1024 * 1024:
                raise ValueError("request body is too large")
            if length == 0:
                return {}
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            return payload

        def _api_error(self, exc: Exception) -> None:
            if isinstance(exc, KeyError):
                self._send_json({"error": str(exc)}, 404)
            elif isinstance(exc, StaleRuntimeError):
                self._send_json({"error": str(exc)}, 409)
            elif isinstance(exc, (ValueError, json.JSONDecodeError)):
                self._send_json({"error": str(exc)}, 400)
            else:
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if not self._guard_api_request(parsed.path):
                return
            if parsed.path in STATIC_FILES:
                filename, content_type = STATIC_FILES[parsed.path]
                self._send_static(filename, content_type)
                return
            if parsed.path == "/api/health":
                self._send_json(runtime.health() if hasattr(runtime, "health") else {"status": "ok"})
                return
            if parsed.path == "/api/token":
                self._send_json({"token": control_token})
                return
            if parsed.path == "/api/overview":
                self._send_json(runtime.overview())
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
                except Exception as exc:
                    self._api_error(exc)
                    return
                self._send_json(payload)
                return
            if parsed.path == "/api/catalog":
                query = parse_qs(parsed.query)
                try:
                    payload = runtime.catalog(
                        (query.get("q") or [""])[0],
                        int((query.get("offset") or ["0"])[0]),
                        int((query.get("limit") or ["30"])[0]),
                    )
                except Exception as exc:
                    self._api_error(exc)
                    return
                self._send_json(payload)
                return
            if parsed.path == "/api/product":
                parent_asin = (parse_qs(parsed.query).get("parent_asin") or [""])[0]
                try:
                    self._send_json(runtime.product(parent_asin))
                except Exception as exc:
                    self._api_error(exc)
                return
            if parsed.path == "/api/documents":
                self._send_json(runtime.documents())
                return
            if parsed.path == "/api/document":
                document_id = (parse_qs(parsed.query).get("id") or [""])[0]
                try:
                    self._send_json(runtime.document(document_id))
                except Exception as exc:
                    self._api_error(exc)
                return
            if parsed.path == "/api/experiments":
                self._send_json(runtime.experiments())
                return
            if parsed.path == "/api/jobs":
                self._send_json(runtime.jobs())
                return
            self._send_json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if not self._guard_api_request(parsed.path):
                return
            try:
                payload = self._read_json()
                if parsed.path == "/api/jobs/evaluation":
                    self._send_json(runtime.start_evaluation(), 202)
                    return
                if parsed.path == "/api/jobs/tests":
                    self._send_json(runtime.start_tests(), 202)
                    return
                if parsed.path == "/api/jobs/generalization":
                    self._send_json(runtime.start_generalization(), 202)
                    return
                if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/cancel"):
                    job_id = parsed.path.removeprefix("/api/jobs/").removesuffix("/cancel")
                    self._send_json(runtime.cancel_job(job_id))
                    return
                if parsed.path == "/api/lab/reset":
                    self._send_json(runtime.lab_reset(payload.get("profile")))
                    return
                if parsed.path == "/api/lab/respond":
                    self._send_json(runtime.lab_respond(
                        str(payload.get("session_id") or ""),
                        str(payload.get("message") or ""),
                    ))
                    return
                if parsed.path == "/api/shutdown":
                    self._send_json({"status": "stopping"})
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                    return
                self._send_json({"error": "not found"}, 404)
            except Exception as exc:
                self._api_error(exc)

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
    parser.add_argument(
        "--rerank-mode",
        choices=("off", "shadow", "active"),
        default=None,
        help=(
            "Agent reranker mode. Defaults to TECHJAM_RERANK_MODE, then off; "
            "the selected mode is fixed for this Workbench process."
        ),
    )
    args = parser.parse_args()

    if args.host not in {"127.0.0.1", "localhost"}:
        parser.error("Layer Observer contains public labels and may bind only to a loopback host")

    print("Loading catalog, public sessions, Agent index, and prior results...")
    project_root = Path(__file__).resolve().parents[1]
    runtime = WorkbenchRuntime.from_paths(
        args.catalog,
        args.dataset,
        args.results,
        project_root=project_root,
        rerank_mode=args.rerank_mode,
    )
    server = ExclusiveHTTPServer((args.host, args.port), make_handler(runtime))
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
        if hasattr(runtime, "close"):
            runtime.close()


if __name__ == "__main__":
    main()
