"""Serve the web visualizer locally.

Run: python -m demo.serve   then open http://localhost:8000/visualizer.html

Serves the demo/web directory (visualizer.html + sample_trajectory.json) over HTTP so the
visualizer can fetch the bundled sample. Generate a fresh trajectory with:

    python -m demo.cli --example hard --export-json demo/web/sample_trajectory.json
"""
import argparse
import functools
import http.server
import os
import socketserver

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Serve the HRM web visualizer.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=WEB_DIR)
    with socketserver.TCPServer(("", args.port), handler) as httpd:
        url = f"http://localhost:{args.port}/visualizer.html"
        print(f"Serving {WEB_DIR}\nOpen {url}  (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
