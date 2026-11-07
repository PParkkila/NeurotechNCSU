"""Local, opt-in SSVEP drawing and validation with a NeuroPawn."""

from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import webbrowser

import numpy as np

from .recording import NeuroPawn
from .ssvep import fit_gate, score_targets, select


INPUTS = (1, 2, 3)
FREQUENCIES = (10.0, 12.0)
SECONDS = 4


def make_handler(board, output):
    page = Path(__file__).with_name("selection.html").read_bytes()
    windows_scored = 0

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def send(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status, result):
            self.send(status, json.dumps(result, allow_nan=False).encode(), "application/json")

        def do_GET(self):
            if self.path == "/":
                self.send(200, page, "text/html; charset=utf-8")
            else:
                self.send_error(404)

        def do_POST(self):
            nonlocal windows_scored
            host = f"127.0.0.1:{self.server.server_port}"
            origin = f"http://{host}"
            if self.headers.get("Host") != host or self.headers.get("Origin", origin) != origin:
                self.send_json(403, {"error": "Use the local app URL printed in Terminal."})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 64_000 or self.headers.get_content_type() != "application/json":
                    raise ValueError("Expected a small JSON request.")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object.")

                if self.path == "/score":
                    eeg = board.capture(SECONDS)
                    scores = score_targets(eeg, board.fs, frequencies=FREQUENCIES)
                    gate = data.get("gate")
                    result = {
                        "scores": scores.tolist(),
                        "forced": int(np.argmax(scores)),
                        "selection": select(scores, gate) if gate is not None else None,
                    }
                    windows_scored += 1
                    print(f"Scored EEG window {windows_scored}.", flush=True)
                elif self.path == "/calibrate":
                    result = {"gate": fit_gate(data["scores"], data["labels"])}
                    print("Calibration passed. Held-out validation can begin.", flush=True)
                elif self.path == "/result":
                    report = {
                        "mode": "ssvep_gaze_selection",
                        "gain": board.gain,
                        "board_inputs": board.inputs,
                        "sample_rate": board.fs,
                        "frequencies": FREQUENCIES,
                        "capture_seconds": SECONDS,
                        "gate": data["gate"],
                        "calibration": data["calibration"],
                        "validation": data["validation"],
                    }
                    output.mkdir(parents=True, exist_ok=True)
                    path = output / f"ssvep-{datetime.now():%Y%m%d-%H%M%S-%f}.json"
                    with path.open("x", encoding="utf-8") as stream:
                        json.dump(report, stream, indent=2, allow_nan=False)
                    print(f"Saved validation report to {path.resolve()}", flush=True)
                    result = {"saved": str(path)}
                else:
                    self.send_error(404)
                    return
                self.send_json(200, result)
            except (BrokenPipeError, ConnectionResetError):
                return
            except (KeyError, TypeError, ValueError, OSError, RuntimeError) as exc:
                print(f"{self.path}: {exc}", flush=True)
                self.send_json(422, {"error": str(exc)})

    return Handler


def run_selection(args):
    from brainflow.board_shim import BoardShim

    BoardShim.disable_board_logger()
    print("Preparing inputs 1–3 for O1, Oz, and O2. This takes about 30 seconds.", flush=True)
    with NeuroPawn(args.port, gain=args.gain, inputs=INPUTS) as board:
        with HTTPServer(("127.0.0.1", 0), make_handler(board, args.out)) as server:
            url = f"http://127.0.0.1:{server.server_port}/"
            print(f"Open {url} to start. Press Ctrl-C in Terminal to stop.", flush=True)
            if args.open:
                webbrowser.open(url)
            server.serve_forever()
