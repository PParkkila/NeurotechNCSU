"""Serve the game and its local NeuroPawn connection over loopback."""

from dataclasses import dataclass
from hashlib import file_digest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Lock
from time import monotonic
from uuid import UUID, uuid4

import numpy as np

from .model import decoder_features
from .personalization import MovePrediction, PlayerDecoder
from .recording import ADCRailError, MOVE_NAMES, NeuroPawn, capture_move


_LOOPBACK = "127.0.0.1"
_SHAPES = tuple(name.lower() for name in MOVE_NAMES)
_MAX_BODY = 4096
_PENDING_LIMIT = 128
_PENDING_SECONDS = 600
_STATIC_TYPES = {
    "index.html": "text/html; charset=utf-8",
    "styles.css": "text/css; charset=utf-8",
    **dict.fromkeys(
        ("api.js", "app.js", "battle.js", "config.js", "input.js", "shapes.js"),
        "text/javascript; charset=utf-8",
    ),
}


class _RequestError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class _ClientDisconnected(Exception):
    pass


def _uuid(value, field):
    try:
        if isinstance(value, str) and len(value) in (32, 36):
            return UUID(value).hex
    except ValueError:
        pass
    raise _RequestError(400, f"{field} must be a UUID.")


@dataclass(slots=True)
class _Round:
    profile: str
    expires: float
    features: np.ndarray | None
    prediction: MovePrediction | None
    actual: int | None = None
    examples: int | None = None


class _Bridge:
    def __init__(self, args, data, board):
        self.board = board
        self.profiles = Path(args.profiles)
        self.seed_features = decoder_features(data["eeg"], data["fs"])
        self.seed_labels = data["labels"]
        with args.session.open("rb") as archive:
            digest = file_digest(archive, "sha256")
        digest.update(bytes([args.imu]))
        self.signature = digest.hexdigest()
        self.rest_seconds = data["baseline"].shape[-1] / data["fs"]
        self.seconds = data["eeg"].shape[-1] / data["fs"]
        self.skip = data["skip"]
        self.capture_lock = Lock()
        self.pending_lock = Lock()
        self.pending = {}

    def _decoder(self, profile):
        return PlayerDecoder(
            self.profiles / f"{profile}.sqlite3", self.seed_features, self.seed_labels,
            signature=self.signature,
        )

    def _prune(self, now):
        expired = [key for key, item in self.pending.items() if item.expires <= now]
        for key in expired:
            del self.pending[key]

    def status(self):
        return {
            "available": True,
            "rest_seconds": self.rest_seconds,
            "imagery_seconds": self.seconds + self.skip,
        }

    def capture(self, profile, emit):
        with self._decoder(profile) as decoder:
            _, imagery = capture_move(
                self.board, seconds=self.seconds, skip=self.skip,
                baseline_seconds=self.rest_seconds, on_phase=lambda phase: emit({"phase": phase}),
            )
            emit({"phase": "decoding"})
            features = decoder_features(imagery[None], self.board.fs)[0]
            prediction = decoder.predict(features)
        round_id = uuid4()
        with self.pending_lock:
            now = monotonic()
            self._prune(now)
            if len(self.pending) >= _PENDING_LIMIT:
                del self.pending[next(iter(self.pending))]
            self.pending[round_id.hex] = _Round(profile, now + _PENDING_SECONDS, features, prediction)
        return {
            "id": str(round_id),
            "shape": _SHAPES[prediction.move],
            "score": prediction.confidence,
            "model": "player" if prediction.using_player else "seed",
            "examples": prediction.examples,
        }

    def confirm(self, round_id, actual):
        with self.pending_lock:
            self._prune(monotonic())
            pending = self.pending.get(round_id)
            if pending is None:
                raise _RequestError(404, "This prediction expired. Capture another round.")
            if pending.actual is not None:
                if pending.actual != actual:
                    raise _RequestError(409, "This round was already confirmed as a different shape.")
                return pending.examples
            with self._decoder(pending.profile) as decoder:
                examples = decoder.teach(round_id, pending.features, actual, pending.prediction)
            pending.actual, pending.examples = actual, examples
            pending.features = pending.prediction = None
            return examples


class _GameServer(ThreadingHTTPServer):
    daemon_threads = False

    def __init__(self, port, bridge, site):
        self.bridge = bridge
        self.assets = {}
        for name, content_type in _STATIC_TYPES.items():
            path = site / name
            if path.is_file():
                self.assets[f"/{name}"] = (content_type, path.read_bytes())
        self.assets["/"] = self.assets["/index.html"]
        super().__init__((_LOOPBACK, port), _Handler)
        self.authority = _LOOPBACK if self.server_port == 80 else f"{_LOOPBACK}:{self.server_port}"
        self.origin = f"http://{self.authority}"


class _Handler(BaseHTTPRequestHandler):
    def setup(self):
        self.request.settimeout(10)
        super().setup()

    def _respond(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, data):
        self._respond(status, "application/json; charset=utf-8", json.dumps(data).encode())

    def _check_host(self):
        if self.headers.get_all("Host") != [self.server.authority]:
            raise _RequestError(403, "Use the loopback address printed by the server.")

    def _body(self):
        if self.headers.get_all("Origin") != [self.server.origin]:
            raise _RequestError(403, "This request must come from the local game page.")
        if self.headers.get_content_type() != "application/json":
            raise _RequestError(415, "Send an application/json request.")
        lengths = self.headers.get_all("Content-Length") or []
        if self.headers.get("Transfer-Encoding") or len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
            raise _RequestError(400, "A valid Content-Length is required.")
        if len(lengths[0]) > len(str(_MAX_BODY)) or int(lengths[0]) > _MAX_BODY:
            raise _RequestError(413, "Request body is too large.")
        try:
            length = int(lengths[0])
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("Incomplete request body.")
            data = json.loads(body)
        except (ValueError, TimeoutError, RecursionError) as exc:
            raise _RequestError(400, "Invalid JSON request.") from exc
        if not isinstance(data, dict):
            raise _RequestError(400, "Expected a JSON object.")
        return data

    def do_GET(self):
        try:
            self._check_host()
            path = self.path.partition("?")[0]
            if path == "/eeg/status":
                self._json(200, self.server.bridge.status())
            elif path in self.server.assets:
                self._respond(200, *self.server.assets[path])
            else:
                self._json(404, {"error": "Not found."})
        except _RequestError as exc:
            self._json(exc.status, {"error": str(exc)})

    def _emit(self, message):
        try:
            self.wfile.write(json.dumps(message).encode() + b"\n")
            self.wfile.flush()
        except OSError as exc:
            raise _ClientDisconnected from exc

    def _capture(self, profile):
        bridge = self.server.bridge
        if not bridge.capture_lock.acquire(blocking=False):
            raise _RequestError(409, "An EEG capture is already running. Try again when it finishes.")
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                self._emit({"prediction": bridge.capture(profile, self._emit)})
            except _ClientDisconnected:
                return
            except ADCRailError:
                self._emit({"error": "EEG clipped. Check the electrode contacts and ear clip, then try again."})
            except Exception as exc:
                self.log_error("EEG capture failed: %s", exc)
                self._emit({"error": "The EEG round failed. Check the headset connection and try again."})
        except (_ClientDisconnected, OSError):
            pass
        finally:
            bridge.capture_lock.release()

    def do_POST(self):
        try:
            self._check_host()
            path = self.path.partition("?")[0]
            if path == "/eeg/capture":
                self._capture(_uuid(self._body().get("profile"), "profile"))
            elif path == "/eeg/confirm":
                data = self._body()
                round_id = _uuid(data.get("id"), "id")
                shape = data.get("shape")
                if shape not in _SHAPES:
                    raise _RequestError(400, "Choose circle, triangle, or square.")
                examples = self.server.bridge.confirm(round_id, _SHAPES.index(shape))
                self._json(200, {"examples": examples})
            else:
                self._json(404, {"error": "Not found."})
        except _RequestError as exc:
            self._json(exc.status, {"error": str(exc)})
        except (RuntimeError, ValueError) as exc:
            self.log_error("EEG request failed: %s", exc)
            self._json(500, {"error": "Could not save the confirmed move to the local profile."})


def run_web(args, data):
    """Run one local board stream and the browser bridge until interrupted."""
    eeg = data["eeg"]
    if data["source"] != "neuropawn" or data["task"] != "rps" or data["baseline"] is None:
        raise ValueError("The browser game requires a three-move NeuroPawn seed with resting EEG.")
    if not data["inputs"] or len(data["inputs"]) != eeg.shape[1] or data["gain"] is None:
        raise ValueError("The seed must identify the board inputs and gain used for training.")

    site = Path(__file__).resolve().parents[1]
    with NeuroPawn(args.port, imu=args.imu, inputs=data["inputs"], gain=data["gain"]) as board:
        if board.fs != data["fs"]:
            raise ValueError("The headset sample rate differs from the seed recording.")
        bridge = _Bridge(args, data, board)
        with _GameServer(args.listen_port, bridge, site) as server:
            print(f"Open {server.origin} for the local NeuroPawn game. Press Ctrl-C to stop.", flush=True)
            server.serve_forever()
