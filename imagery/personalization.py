"""Learn a player's imagery decoder from explicitly reported moves."""

from __future__ import annotations

from contextlib import closing
from dataclasses import asdict, dataclass
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


_CLASSES = (0, 1, 2)
_COMPONENTS = 5
_SEED_MASS = 18.0
_FEEDBACK_WEIGHT = 0.25
_BLEND_PRIOR = 48
_BLEND_STEP = 0.05
_POLICY_VERSION = 2
_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class MovePrediction:
    move: int
    confidence: float
    seed_move: int
    seed_confidence: float
    examples: int
    candidate_move: int
    candidate_confidence: float
    using_player: bool
    policy_version: int = _POLICY_VERSION
    player_weight: float = 0.0


@dataclass(frozen=True, slots=True)
class _FittedDecoder:
    scaler: StandardScaler
    components: np.ndarray
    classifier: LogisticRegression

    def project(self, features: np.ndarray) -> np.ndarray:
        return self.scaler.transform(features) @ self.components.T

    def scores(self, features: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(self.project(features[None]))[0]


def _fit_seed(features, labels):
    scaler = StandardScaler().fit(features)
    scaled = scaler.transform(features)
    count = min(_COMPONENTS, len(features) - 1, features.shape[1])
    components = PCA(n_components=count, svd_solver="full").fit(scaled).components_
    classifier = LogisticRegression().fit(scaled @ components.T, labels)
    return _FittedDecoder(scaler, components, classifier)


def _fit_player(seed, seed_features, seed_labels, player_features, actual):
    if not len(actual):
        return seed
    features = np.concatenate((seed_features, player_features))
    labels = np.concatenate((seed_labels, np.asarray(actual, dtype=np.uint8)))
    seed_counts = np.bincount(seed_labels, minlength=len(_CLASSES))
    weights = np.concatenate((
        np.full(len(seed_labels), _SEED_MASS / len(seed_labels)),
        np.full(len(actual), _FEEDBACK_WEIGHT),
    ))
    # Preserve the seed's class masses and regularization scale throughout replay.
    class_mass = np.bincount(labels, weights=weights, minlength=len(_CLASSES))
    weights *= seed_counts[labels] / class_mass[labels]
    classifier = LogisticRegression().fit(seed.project(features), labels, sample_weight=weights)
    return _FittedDecoder(seed.scaler, seed.components, classifier)


def _best_move(scores):
    best = int(np.argmax(scores))
    return _CLASSES[best], float(scores[best])


def _adaptation_evidence(examples) -> tuple[int, bool]:
    """Use predictions recorded before feedback under the current learning policy."""
    counts = [0] * len(_CLASSES)
    gain = [0] * len(_CLASSES)
    for _, actual, metadata in examples:
        prediction = json.loads(metadata)
        if prediction.get("policy_version") != _POLICY_VERSION:
            continue
        counts[actual] += 1
        gain[actual] += int(prediction["candidate_move"] == actual) - int(prediction["seed_move"] == actual)
    total = sum(counts)
    if total < 12 or min(counts) < 3 or sum(gain) < 2:
        return total, False
    balanced_gain = sum(Fraction(hits, count) for hits, count in zip(gain, counts)) / len(_CLASSES)
    return total, balanced_gain >= Fraction(1, 10)


def _adaptation_state(examples) -> tuple[int, float]:
    count, eligible = _adaptation_evidence(examples)
    previous = json.loads(examples[-1][2]) if examples else {}
    prior_weight = previous.get("weight_before_feedback", 0.0)
    target = count / (count + _BLEND_PRIOR) if eligible else 0.0
    weight = min(max(target, prior_weight - _BLEND_STEP), prior_weight + _BLEND_STEP)
    return count, weight


def _predict_move(seed_model, player_model, row, examples, policy_examples, player_weight):
    seed_scores = seed_model.scores(row)
    seed = _best_move(seed_scores)
    candidate = displayed = seed
    if policy_examples:
        blend = policy_examples / (policy_examples + _BLEND_PRIOR)
        player_delta = player_model.scores(row) - seed_scores
        candidate = _best_move(seed_scores + blend * player_delta)
        displayed = _best_move(seed_scores + player_weight * player_delta)
    move, confidence = displayed
    return MovePrediction(
        move=move, confidence=confidence, seed_move=seed[0], seed_confidence=seed[1],
        examples=examples, candidate_move=candidate[0], candidate_confidence=candidate[1],
        using_player=player_weight > 0, player_weight=player_weight,
    )


class PlayerDecoder:
    """Store confirmed feature rows locally and keep the original decoder fixed."""

    def __init__(self, profile: Path, seed_features, seed_labels, *, signature: str):
        features = np.asarray(seed_features)
        labels = np.asarray(seed_labels)
        if (
            features.ndim != 2
            or features.shape[1] == 0
            or features.dtype.kind not in "iuf"
            or not np.isfinite(features).all()
        ):
            raise ValueError("Seed features must be a matrix of finite feature rows.")
        if (
            labels.shape != (len(features),)
            or labels.dtype.kind not in "iuf"
            or not np.array_equal(np.unique(labels), _CLASSES)
        ):
            raise ValueError("Seed examples must include all three moves.")
        if not isinstance(signature, str) or not signature.strip():
            raise ValueError("A calibration signature is required for the player profile.")

        self._profile = Path(profile).resolve()
        self._seed_features = np.array(features, dtype="<f8", order="C", copy=True)
        self._seed_labels = np.array(labels, dtype=np.uint8, copy=True)
        self._signature = signature
        digest = sha256(self._seed_features.tobytes())
        digest.update(self._seed_labels.tobytes())
        self._seed_hash = digest.hexdigest()
        self._seed_model = None
        self._model = None
        self._examples = 0
        self._policy_examples = 0
        self._player_weight = 0.0

    def __enter__(self) -> PlayerDecoder:
        if self._model is not None:
            raise RuntimeError("This player decoder is already open.")
        self._seed_model = self._model = _fit_seed(self._seed_features, self._seed_labels)
        try:
            self._refresh()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._seed_model = self._model = None
        self._examples = 0
        self._policy_examples = 0
        self._player_weight = 0.0

    def _row(self, features) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Open the player decoder using a with block.")
        row = np.asarray(features)
        if (
            row.shape != self._seed_features.shape[1:]
            or row.dtype.kind not in "iuf"
            or not np.isfinite(row).all()
        ):
            raise ValueError("Expected one finite feature row matching the calibration.")
        return row.astype("<f8", copy=False)

    def _check_profile(self, database, *, initialize=False) -> bool:
        version = database.execute("PRAGMA user_version").fetchone()[0]
        if version == 0 and not database.execute("SELECT name FROM sqlite_master").fetchone():
            if not initialize:
                return False
            database.execute("CREATE TABLE identity (signature TEXT NOT NULL, seed_hash TEXT NOT NULL)")
            database.execute(
                """CREATE TABLE examples (
                    round_id TEXT PRIMARY KEY NOT NULL,
                    features BLOB NOT NULL,
                    actual INTEGER NOT NULL CHECK (actual BETWEEN 0 AND 2),
                    prediction TEXT NOT NULL
                )"""
            )
            database.execute("INSERT INTO identity VALUES (?, ?)", (self._signature, self._seed_hash))
            database.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            return True
        if version != _SCHEMA_VERSION:
            raise ValueError(f"Unsupported player profile format in {self._profile}.")
        identity = database.execute("SELECT signature, seed_hash FROM identity").fetchall()
        if identity != [(self._signature, self._seed_hash)]:
            raise ValueError("This player profile belongs to a different calibration. Choose another profile.")
        return True

    def _fit_examples(self, examples):
        if not examples:
            return self._seed_model
        rows, confirmed, _ = zip(*examples)
        player_features = np.stack([np.frombuffer(row, dtype="<f8") for row in rows])
        return _fit_player(self._seed_model, self._seed_features, self._seed_labels, player_features, confirmed)

    def _read_examples(self, database):
        return database.execute("SELECT features, actual, prediction FROM examples ORDER BY rowid").fetchall()

    def _profile_state(self, examples):
        if len(examples) == self._examples:
            return self._model, self._policy_examples, self._player_weight
        count, player_weight = _adaptation_state(examples)
        return self._fit_examples(examples), count, player_weight

    def _refresh(self) -> None:
        if not self._profile.exists():
            self._model, self._examples = self._seed_model, 0
            self._policy_examples = 0
            self._player_weight = 0.0
            return
        try:
            uri = f"{self._profile.as_uri()}?mode=ro"
            with closing(sqlite3.connect(uri, uri=True)) as database, database:
                database.execute("BEGIN")
                examples = self._read_examples(database) if self._check_profile(database) else []
        except sqlite3.Error as exc:
            raise RuntimeError(f"Could not read player profile {self._profile}. {exc}") from exc
        self._model, self._policy_examples, self._player_weight = self._profile_state(examples)
        self._examples = len(examples)

    def predict(self, features_1d) -> MovePrediction:
        row = self._row(features_1d)
        self._refresh()
        return _predict_move(
            self._seed_model, self._model, row, self._examples, self._policy_examples, self._player_weight,
        )

    def teach(self, round_id: str, features_1d, actual: int, prediction: MovePrediction) -> int:
        """Add a confirmed move once and refit from the saved rows."""
        row = self._row(features_1d).tobytes()
        if not isinstance(round_id, str) or not round_id.strip():
            raise ValueError("A round ID is required to teach the player decoder.")
        if isinstance(actual, bool) or not isinstance(actual, (int, np.integer)) or actual not in _CLASSES:
            raise ValueError("A confirmed move must be 0, 1, or 2.")
        if not isinstance(prediction, MovePrediction):
            raise ValueError("Teaching requires the prediction made before the move was reported.")
        actual = int(actual)
        self._profile.parent.mkdir(parents=True, exist_ok=True)
        try:
            with closing(sqlite3.connect(self._profile)) as database, database:
                database.execute("BEGIN IMMEDIATE")
                self._check_profile(database, initialize=True)
                existing = database.execute(
                    "SELECT features, actual FROM examples WHERE round_id = ?", (round_id,)
                ).fetchone()
                if existing is not None and existing != (row, actual):
                    raise ValueError("This round ID already has a different confirmed move or feature row.")
                examples = self._read_examples(database)
                if existing is None:
                    # A confirmation may arrive after another round has already taught this profile.
                    _, prior_weight = _adaptation_state(examples)
                    metadata = json.dumps(dict(asdict(prediction), weight_before_feedback=prior_weight), allow_nan=False)
                    database.execute(
                        "INSERT INTO examples VALUES (?, ?, ?, ?)", (round_id, row, actual, metadata)
                    )
                    examples.append((row, actual, metadata))
                model, policy_examples, player_weight = self._profile_state(examples)
        except sqlite3.Error as exc:
            raise RuntimeError(f"Could not save player profile {self._profile}. {exc}") from exc
        self._model, self._policy_examples, self._player_weight = model, policy_examples, player_weight
        self._examples = len(examples)
        return self._examples
