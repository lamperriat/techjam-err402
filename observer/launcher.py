from __future__ import annotations

import hashlib
import json
import traceback
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

OBSERVER_URL = "http://127.0.0.1:8765"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_id() -> str:
    return hashlib.sha256(str(PROJECT_ROOT).casefold().encode("utf-8")).hexdigest()[:16]


def _running_project() -> str | None:
    try:
        with urllib.request.urlopen(f"{OBSERVER_URL}/api/health", timeout=1.2) as response:
            payload = json.load(response)
            if response.status == 200 and payload.get("trace_schema"):
                return str(payload.get("project_id") or "unknown-project")
            return None
    except (OSError, ValueError, urllib.error.URLError):
        return None


def _show_error(message: str) -> None:
    try:
        from tkinter import messagebox

        messagebox.showerror("Agent Workbench could not start", message)
    except Exception:
        pass


def main() -> None:
    log_path = PROJECT_ROOT / "observer_startup_error.log"
    try:
        log_path.unlink(missing_ok=True)
        running_project = _running_project()
        if running_project == _project_id():
            webbrowser.open(OBSERVER_URL)
            return
        if running_project is not None:
            raise RuntimeError(
                "Port 8765 is already used by a different Agent Workbench project. "
                "Stop that instance before starting this repository."
            )
        from observer.server import main as serve

        serve()
    except Exception as exc:
        log_path.write_text(traceback.format_exc(), encoding="utf-8")
        _show_error(f"{type(exc).__name__}: {exc}\n\nDetails: {log_path}")


if __name__ == "__main__":
    main()
