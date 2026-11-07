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
_SEED_MASS_PER_CLASS = 6.0
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


@dataclass(frozen=True, slots=True)
class _FittedDecoder:
    scaler: StandardScaler
    components: np.ndarray
    classifier: LogisticRegression

    def predict(self, features: np.ndarray) -> tuple[int, float]:
        projected = self.scaler.transform(features[None]) @ self.components.T
        scores = self.classifier.predict_proba(projected)[0]
        best = int(np.argmax(scores))
        return int(self.classifier.classes_[best]), float(scores[best])


def _fit(features, labels, weights=None):
    scaler = StandardScaler().fit(features, sample_weight=weights)
    scaled = scaler.transform(features)
    count = min(_COMPONENTS, len(features) - 1, features.shape[1])
    if weights is None:
        components = PCA(n_components=count, svd_solver="full").fit(scaled).components_
    else:
        # The weighted scaler centers these rows before the weighted SVD.
        _, _, vectors = np.linalg.svd(scaled * np.sqrt(weights[:, None]), full_matrices=False)
        components = vectors[:count]
    classifier = LogisticRegression().fit(scaled @ components.T, labels, sample_weight=weights)
    return _FittedDecoder(scaler, components, classifier)


def _earned_adaptation(examples) -> bool:
    """Compare the two predictions captured before each reported move."""
    if len(examples) < 12:
        return False
    counts = [0] * len(_CLASSES)
    gain = [0] * len(_CLASSES)
    for _, actual, metadata in examples:
        prediction = json.loads(metadata)
        counts[actual] += 1
        gain[actual] += int(prediction["candidate_move"] == actual) - int(prediction["seed_move"] == actual)
    if min(counts) < 3 or sum(gain) < 2:
        return False
    balanced_gain = sum(Fraction(hits, count) for hits, count in zip(gain, counts)) / len(_CLASSES)
    return balanced_gain >= Fraction(1, 10)


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
        self._using_player = False

    def __enter__(self) -> PlayerDecoder:
        if self._model is not None:
            raise RuntimeError("This player decoder is already open.")
        self._seed_model = self._model = _fit(self._seed_features, self._seed_labels)
        try:
            self._refresh()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._seed_model = self._model = None
        self._examples = 0
        self._using_player = False

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
        features = np.concatenate((self._seed_features, player_features))
        labels = np.concatenate((self._seed_labels, np.array(confirmed, dtype=np.uint8)))

        seed_counts = np.bincount(self._seed_labels, minlength=len(_CLASSES))
        weights = np.concatenate((_SEED_MASS_PER_CLASS / seed_counts[self._seed_labels], np.ones(len(rows))))
        # Give the classes equal total weight without changing their internal seed/player mix.
        class_mass = np.bincount(labels, weights=weights, minlength=len(_CLASSES))
        weights *= class_mass.mean() / class_mass[labels]
        return _fit(features, labels, weights)

    def _read_examples(self, database):
        return database.execute("SELECT features, actual, prediction FROM examples ORDER BY rowid").fetchall()

    def _profile_state(self, examples):
        if len(examples) == self._examples:
            return self._model, self._using_player
        return self._fit_examples(examples), _earned_adaptation(examples)

    def _refresh(self) -> None:
        if not self._profile.exists():
            self._model, self._examples = self._seed_model, 0
            self._using_player = False
            return
        try:
            uri = f"{self._profile.as_uri()}?mode=ro"
            with closing(sqlite3.connect(uri, uri=True)) as database, database:
                database.execute("BEGIN")
                examples = self._read_examples(database) if self._check_profile(database) else []
        except sqlite3.Error as exc:
            raise RuntimeError(f"Could not read player profile {self._profile}. {exc}") from exc
        self._model, self._using_player = self._profile_state(examples)
        self._examples = len(examples)

    def predict(self, features_1d) -> MovePrediction:
        row = self._row(features_1d)
        self._refresh()
        seed = self._seed_model.predict(row)
        candidate = seed if self._model is self._seed_model else self._model.predict(row)
        move, confidence = candidate if self._using_player else seed
        return MovePrediction(
            move=move, confidence=confidence, seed_move=seed[0], seed_confidence=seed[1],
            examples=self._examples, candidate_move=candidate[0], candidate_confidence=candidate[1],
            using_player=self._using_player,
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
        metadata = json.dumps(asdict(prediction), allow_nan=False)
        self._profile.parent.mkdir(parents=True, exist_ok=True)
        try:
            with closing(sqlite3.connect(self._profile)) as database, database:
                database.execute("BEGIN IMMEDIATE")
                self._check_profile(database, initialize=True)
                existing = database.execute(
                    "SELECT features, actual FROM examples WHERE round_id = ?", (round_id,)
                ).fetchone()
                if existing is None:
                    database.execute(
                        "INSERT INTO examples VALUES (?, ?, ?, ?)", (round_id, row, actual, metadata)
                    )
                elif existing != (row, actual):
                    raise ValueError("This round ID already has a different confirmed move or feature row.")
                examples = self._read_examples(database)
                model, using_player = self._profile_state(examples)
        except sqlite3.Error as exc:
            raise RuntimeError(f"Could not save player profile {self._profile}. {exc}") from exc
        self._model, self._using_player = model, using_player
        self._examples = len(examples)
        return self._examples
