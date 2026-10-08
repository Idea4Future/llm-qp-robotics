#!/usr/bin/env python3
"""Serve the local chat UI; Python standard library, no Isaac/LLM in this process."""
from __future__ import annotations
import argparse
from pathlib import Path
import signal
import sys
import threading
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from opti_web.fetch_assets import fetch, ready
from opti_web.server import ChatServer


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--fetch-assets", action="store_true", help="Download pinned official KaTeX browser assets")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("port must be 0..65535")
    if args.fetch_assets:
        fetch()
    if not ready():
        print("KaTeX assets missing. Run again with --fetch-assets for local math rendering.", file=sys.stderr)
    server = ChatServer(args.port, ROOT)

    def stop(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Chat UI: {url}", flush=True)
    if args.open_browser:
        webbrowser.open(url, new=2)
    try:
        server.serve_forever(poll_interval=.2)
    finally:
        server.manager.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
