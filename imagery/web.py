"""Serve the game and its local NeuroPawn connection over loopback."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

from .recording import ADCRailError, MOVE_NAMES, NeuroPawn, capture_move
from .web_session import IMAGERY_SECONDS, REST_SECONDS, GameSession


_LOOPBACK = "127.0.0.1"
_SHAPES = tuple(name.lower() for name in MOVE_NAMES)
_MAX_BODY = 4096
_STATIC_TYPES = {
    "index.html": "text/html; charset=utf-8",
    "styles.css": "text/css; charset=utf-8",
    **dict.fromkeys(
        ("api.js", "app.js", "battle.js", "config.js", "input.js", "shapes.js", "session-game.js",
         "browser-serial.js", "browser-decoder.js", "browser-store.js", "browser-game.js"),
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


class _Bridge:
    def __init__(self, args, board):
        self.board = board
        self.directory = args.sessions
        self.configuration = dict(fs=board.fs, inputs=args.inputs, gain=args.gain, imu=args.imu)
        self.lock = Lock()

    @contextmanager
    def operation(self):
        if not self.lock.acquire(blocking=False):
            raise _RequestError(409, "Another EEG action is in progress. Try again when it finishes.")
        try:
            yield
        finally:
            self.lock.release()

    def status(self):
        return {
            "available": True,
            "mode": "calibration",
            "rest_seconds": REST_SECONDS + 0.5,
            "imagery_seconds": IMAGERY_SECONDS,
        }

    def create(self, player):
        if not isinstance(player, str) or len(player.strip()) > 24:
            raise _RequestError(400, "Use a player name up to 24 characters.")
        return GameSession.create(self.directory, player.strip() or "Player", **self.configuration)

    def open(self, session_id):
        path = self.directory / f"{session_id}.npz"
        try:
            return GameSession.load(path, **self.configuration)
        except FileNotFoundError as exc:
            raise _RequestError(404, "Session not found. Start a new game.") from exc

    def capture(self, game, emit):
        baseline, imagery = capture_move(
            self.board, seconds=IMAGERY_SECONDS, skip=0, baseline_seconds=REST_SECONDS,
            on_phase=lambda phase: emit({"phase": phase}),
        )
        emit({"phase": "decoding"})
        return game.capture(baseline, imagery)


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
            url = urlsplit(self.path)
            bridge = self.server.bridge
            if url.path == "/eeg/status":
                self._json(200, bridge.status())
            elif url.path == "/eeg/session":
                session_ids = parse_qs(url.query).get("session", [])
                session_id = _uuid(session_ids[0] if len(session_ids) == 1 else None, "session")
                self._json(200, bridge.open(session_id).state())
            elif url.path in self.server.assets:
                self._respond(200, *self.server.assets[url.path])
            else:
                self._json(404, {"error": "Not found."})
        except _RequestError as exc:
            self._json(exc.status, {"error": str(exc)})
        except ValueError as exc:
            self._json(409, {"error": str(exc)})
        except OSError as exc:
            self.log_error("EEG session read failed: %s", exc)
            self._json(500, {"error": "The local game session could not be read."})

    def _emit(self, message):
        try:
            self.wfile.write(json.dumps(message).encode() + b"\n")
            self.wfile.flush()
        except OSError as exc:
            raise _ClientDisconnected from exc

    def _capture(self, data):
        bridge = self.server.bridge
        with bridge.operation():
            game = bridge.open(_uuid(data.get("session"), "session"))
            state = game.state()
            question = data.get("question")
            if state["phase"] == "calibration":
                if type(question) is not int or question != state["calibration"]["completed"]:
                    raise _RequestError(409, "The calibration prompt changed. Refresh the session and try again.")
            elif question is not None or state["phase"] not in ("ready", "result"):
                raise _RequestError(409, "Finish the current round before starting another read.")

            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    self._emit({"state": bridge.capture(game, self._emit)})
                except _ClientDisconnected:
                    return
                except ADCRailError:
                    self._emit({"error": "EEG clipped. Check the electrode contacts and ear clip, then try again."})
                except Exception as exc:
                    self.log_error("EEG capture failed: %s", exc)
                    self._emit({"error": "The EEG read failed. Check the headset and local session folder, then try again."})
            except (_ClientDisconnected, OSError):
                return

    def do_POST(self):
        try:
            self._check_host()
            path = urlsplit(self.path).path
            bridge = self.server.bridge
            if path == "/eeg/session":
                self._json(201, bridge.create(self._body().get("player", "")).state())
            elif path == "/eeg/capture":
                self._capture(self._body())
            elif path in ("/eeg/confirm", "/eeg/battle", "/eeg/discard"):
                data = self._body()
                round_id = _uuid(data.get("id"), "id")
                with bridge.operation():
                    game = bridge.open(_uuid(data.get("session"), "session"))
                    if path == "/eeg/confirm":
                        shape = data.get("shape")
                        if shape not in _SHAPES:
                            raise _RequestError(400, "Choose circle, triangle, or square.")
                        state = game.confirm(round_id, shape)
                    elif path == "/eeg/battle":
                        state = game.battle(round_id)
                    else:
                        state = game.discard(round_id)
                self._json(200, state)
            else:
                self._json(404, {"error": "Not found."})
        except _RequestError as exc:
            self._json(exc.status, {"error": str(exc)})
        except ValueError as exc:
            self._json(409, {"error": str(exc)})
        except (RuntimeError, OSError) as exc:
            self.log_error("EEG request failed: %s", exc)
            self._json(500, {"error": "The local game session could not be saved."})


def run_web(args):
    """Run one local board stream and the browser bridge until interrupted."""
    site = Path(__file__).resolve().parents[1]
    with NeuroPawn(args.port, imu=args.imu, inputs=args.inputs, gain=args.gain) as board:
        bridge = _Bridge(args, board)
        with _GameServer(args.listen_port, bridge, site) as server:
            print(f"Open {server.origin} for the local NeuroPawn game. Press Ctrl-C to stop.", flush=True)
            server.serve_forever()
