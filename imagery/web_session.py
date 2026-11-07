"""A local, resumable game with per-person calibration and paired EEG."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from secrets import SystemRandom, randbelow
from tempfile import NamedTemporaryFile
from uuid import UUID, uuid4

import numpy as np

from .model import decoder_features
from .personalization import _adaptation_state, _fit_player, _fit_seed, _predict_move
from .recording import ADCRailError, MOVE_NAMES, adc_rail_mask


_SHAPES = tuple(name.lower() for name in MOVE_NAMES)
_CALIBRATION_COUNT = 20
_VERSION = 1
REST_SECONDS = 2
IMAGERY_SECONDS = 5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _board_setup(fs, inputs, gain, imu) -> dict:
    return {"fs": int(fs), "inputs": [int(value) for value in inputs], "gain": int(gain), "imu": bool(imu)}


def _metrics(confusion: np.ndarray) -> dict:
    hits = np.diag(confusion)
    total = int(confusion.sum())
    correct = int(hits.sum())
    actual, predicted = confusion.sum(axis=1), confusion.sum(axis=0)
    recall = np.divide(hits, actual, out=np.zeros(len(_SHAPES)), where=actual > 0)
    precision = np.divide(hits, predicted, out=np.zeros(len(_SHAPES)), where=predicted > 0)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "macro_recall": float(recall.mean()),
        "macro_precision": float(precision.mean()),
    }


class GameSession:
    """Store one player's calibration, rounds, and labels in one local archive."""

    def __init__(self, path, metadata, baseline, eeg):
        self.path = Path(path)
        self._metadata = metadata
        self._baseline = baseline
        self._eeg = eeg

    @property
    def id(self) -> str:
        return self._metadata["id"]

    @classmethod
    def create(cls, directory, player, *, fs, inputs, gain, imu) -> GameSession:
        if not isinstance(player, str) or not player.strip() or len(player.strip()) > 80:
            raise ValueError("Enter a player name of at most 80 characters.")
        setup = _board_setup(fs, inputs, gain, imu)
        schedule = [shape for shape in range(len(_SHAPES)) for _ in range(7)]
        del schedule[7 * randbelow(len(_SHAPES))]
        SystemRandom().shuffle(schedule)

        session_id = uuid4().hex
        path = Path(directory) / f"{session_id}.npz"
        metadata = {
            "version": _VERSION, "id": session_id, "player": player.strip(), "board": setup,
            "created_at": _now(), "shapes": list(_SHAPES), "calibration_schedule": schedule, "trials": [],
        }
        shape = (0, len(setup["inputs"]))
        baseline = np.empty((*shape, REST_SECONDS * setup["fs"]))
        eeg = np.empty((*shape, IMAGERY_SECONDS * setup["fs"]))
        session = cls(path, metadata, baseline, eeg)
        path.parent.mkdir(parents=True, exist_ok=True)
        session._commit(metadata)
        return session

    @classmethod
    def load(cls, path, *, fs, inputs, gain, imu) -> GameSession:
        path = Path(path)
        try:
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(archive["metadata"].item())
                baseline, eeg = archive["baseline"], archive["eeg"]
            if metadata["version"] != _VERSION or metadata["id"] != UUID(path.stem).hex:
                raise ValueError("Unsupported session format or ID.")
            if metadata["board"] != _board_setup(fs, inputs, gain, imu):
                raise ValueError("The saved session uses a different headset configuration.")
            schedule = metadata["calibration_schedule"]
            if len(schedule) != _CALIBRATION_COUNT or sorted(schedule.count(i) for i in range(3)) != [6, 7, 7]:
                raise ValueError("Invalid saved calibration schedule.")
            trials = metadata["trials"]
            shape = (len(trials), len(metadata["board"]["inputs"]))
            if baseline.shape != (*shape, REST_SECONDS * fs) or eeg.shape != (*shape, IMAGERY_SECONDS * fs):
                raise ValueError("Saved EEG does not match the session metadata.")
        except (KeyError, TypeError) as exc:
            raise ValueError("Invalid saved game session.") from exc
        return cls(path, metadata, baseline, eeg)

    def _commit(self, metadata, baseline=None, imagery=None):
        rest, eeg = self._baseline, self._eeg
        if baseline is not None:
            rest = np.concatenate((rest, baseline[None]))
            eeg = np.concatenate((eeg, imagery[None]))
        file = NamedTemporaryFile(dir=self.path.parent, suffix=".tmp", delete=False)
        temporary = Path(file.name)
        try:
            with file:
                np.savez_compressed(
                    file, baseline=rest, eeg=eeg, metadata=json.dumps(metadata, allow_nan=False),
                )
                file.flush()
                os.fsync(file.fileno())
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)
        self._metadata, self._baseline, self._eeg = metadata, rest, eeg

    def _trials(self):
        return self._metadata["trials"]

    def _current(self):
        trials = self._trials()
        return trials[-1] if len(trials) > _CALIBRATION_COUNT else None

    def _get_round(self, round_id):
        try:
            round_id = UUID(round_id).hex
        except (AttributeError, ValueError) as exc:
            raise ValueError("Invalid game round ID.") from exc
        current = self._current()
        if current is None or current["id"] != round_id:
            raise ValueError("This round is no longer current.")
        return current

    def state(self) -> dict:
        trials = self._trials()
        completed = min(_CALIBRATION_COUNT, len(trials))
        prompt = _SHAPES[self._metadata["calibration_schedule"][completed]] if completed < _CALIBRATION_COUNT else None
        current = self._current()
        round_state = None
        phase = "calibration" if prompt is not None else "ready"
        if current is not None and current["status"] != "discarded":
            phase = current["status"]
            prediction = current["prediction"]
            actual = current["actual"]
            round_state = {
                "id": current["id"], "predicted": _SHAPES[prediction["move"]],
                "score": prediction["confidence"], "model": "player" if prediction["using_player"] else "seed",
                "examples": prediction["examples"], "confirmed": _SHAPES[actual] if actual is not None else None,
                "computer": _SHAPES[current["computer"]] if phase == "result" else None,
            }

        confusion = np.zeros((len(_SHAPES), len(_SHAPES)), dtype=int)
        history, results = [], []
        for trial in trials[_CALIBRATION_COUNT:]:
            actual = trial["actual"]
            if actual is None:
                continue
            confusion[actual, trial["prediction"]["move"]] += 1
            history.append({"round": len(history) + 1, **_metrics(confusion)})
            if trial["status"] == "result":
                results.append({"player": _SHAPES[actual], "computer": _SHAPES[trial["computer"]]})
        return {
            "id": self.id, "player": self._metadata["player"], "file": self.path.name,
            "phase": phase, "calibration": {"completed": completed, "total": _CALIBRATION_COUNT, "prompt": prompt},
            "round": round_state, "metrics": {**_metrics(confusion), "history": history}, "results": results,
        }

    def _epoch(self, samples, seconds):
        board = self._metadata["board"]
        expected = (len(board["inputs"]), seconds * board["fs"])
        samples = np.asarray(samples)
        if samples.shape != expected or samples.dtype.kind not in "iuf" or not np.isfinite(samples).all():
            raise ValueError("Captured EEG does not match the session's channel count or duration.")
        if np.any(adc_rail_mask(samples)):
            raise ADCRailError("EEG clipped. Check the electrodes and try again.")
        if np.any(np.ptp(samples, axis=1) == 0):
            raise ValueError("Captured EEG contains a flat channel.")
        return samples

    def _predict(self, imagery):
        fs = self._metadata["board"]["fs"]
        seed_features = decoder_features(self._eeg[:_CALIBRATION_COUNT, :, fs:], fs)
        seed_labels = np.asarray(self._metadata["calibration_schedule"])
        seed = _fit_seed(seed_features, seed_labels)
        prior = [(i, trial) for i, trial in enumerate(self._trials()) if trial.get("actual") is not None]
        player = seed
        history = []
        if prior:
            rows, records = zip(*prior)
            player_features = decoder_features(self._eeg[list(rows), :, fs:], fs)
            actual = [trial["actual"] for trial in records]
            player = _fit_player(seed, seed_features, seed_labels, player_features, actual)
            for label, trial in zip(actual, records):
                prediction = trial["prediction"]
                metadata = dict(prediction, weight_before_feedback=prediction["player_weight"])
                history.append((None, label, json.dumps(metadata)))
        count, weight = _adaptation_state(history)
        row = decoder_features(imagery[None, :, fs:], fs)[0]
        return _predict_move(seed, player, row, len(history), count, weight)

    def capture(self, baseline, imagery) -> dict:
        current = self._current()
        if current is not None and current["status"] in ("review", "confirmed"):
            raise ValueError("Finish or discard the current round before capturing another.")
        baseline = self._epoch(baseline, REST_SECONDS)
        imagery = self._epoch(imagery, IMAGERY_SECONDS)
        metadata = deepcopy(self._metadata)
        trials = metadata["trials"]
        recorded = _now()
        if len(trials) < _CALIBRATION_COUNT:
            target = metadata["calibration_schedule"][len(trials)]
            trials.append({"kind": "calibration", "label": target, "captured_at": recorded})
        else:
            computer = randbelow(len(_SHAPES))
            prediction = self._predict(imagery)
            trials.append({
                "kind": "game", "id": uuid4().hex, "status": "review", "actual": None,
                "computer": computer, "prediction": asdict(prediction), "captured_at": recorded,
            })
        self._commit(metadata, baseline, imagery)
        return self.state()

    def confirm(self, round_id, shape) -> dict:
        if shape not in _SHAPES:
            raise ValueError("Choose circle, triangle, or square.")
        actual = _SHAPES.index(shape)
        current = self._get_round(round_id)
        if current["actual"] is not None:
            if current["actual"] != actual:
                raise ValueError("This round was already confirmed as a different shape.")
            return self.state()
        if current["status"] != "review":
            raise ValueError("Only a pending prediction can be confirmed.")
        metadata = deepcopy(self._metadata)
        metadata["trials"][-1].update(status="confirmed", actual=actual, confirmed_at=_now())
        self._commit(metadata)
        return self.state()

    def battle(self, round_id) -> dict:
        current = self._get_round(round_id)
        if current["status"] == "result":
            return self.state()
        if current["status"] != "confirmed":
            raise ValueError("Confirm your shape before battling.")
        metadata = deepcopy(self._metadata)
        metadata["trials"][-1].update(status="result", revealed_at=_now())
        self._commit(metadata)
        return self.state()

    def discard(self, round_id) -> dict:
        current = self._get_round(round_id)
        if current["status"] == "discarded":
            return self.state()
        if current["status"] != "review":
            raise ValueError("Only an unconfirmed round can be discarded.")
        metadata = deepcopy(self._metadata)
        metadata["trials"][-1].update(status="discarded", discarded_at=_now())
        self._commit(metadata)
        return self.state()
