"""Port helpers — avoid WinError 10048 when 8050 is already taken."""
from __future__ import annotations

import socket
import urllib.error
import urllib.request

DEFAULT_PORT = 8050
PORT_RANGE = range(8050, 8060)


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def server_alive(port: int, host: str = "127.0.0.1", timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/health", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def resolve_port(preferred: int = DEFAULT_PORT) -> int:
    """Pick a port: reuse preferred if free, or if a healthy server is already there."""
    if not port_in_use(preferred):
        return preferred
    if server_alive(preferred):
        print(f"\n  ARGO SENTINEL is already running at http://127.0.0.1:{preferred}")
        print("  Open that URL in your browser. Stop the other process to restart.\n")
        raise SystemExit(0)
    for port in PORT_RANGE:
        if port == preferred:
            continue
        if not port_in_use(port):
            print(f"  Port {preferred} is busy (stale process). Using port {port} instead.")
            return port
    raise RuntimeError(f"No free port in {PORT_RANGE.start}–{PORT_RANGE.stop - 1}")
