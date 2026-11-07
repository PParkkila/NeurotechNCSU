"""Spectral features and block-held-out comparison for imagery EEG."""

from numbers import Integral

import numpy as np
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BANDS = ((4, 8), (8, 13), (13, 20), (20, 30))
POSTERIOR_CHANNEL_COUNT = 6


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return int(value)


def spectral_features(eeg, fs):
    """Return log band powers in channel-major order, one row per EEG trial."""
    fs = _integer(fs, "fs", 60)
    eeg = np.asarray(eeg)
    if eeg.ndim != 3 or not eeg.shape[0] or not eeg.shape[1]:
        raise ValueError("eeg must have shape (trials, channels, samples)")
    if eeg.shape[2] < fs:
        raise ValueError("each EEG trial must contain at least one second of samples")
    if eeg.dtype.kind not in "iuf" or not np.isfinite(eeg).all():
        raise ValueError("EEG samples must be real and finite")

    frequencies, density = welch(
        eeg.astype(float, copy=False),
        fs=fs,
        window="hann",
        nperseg=fs,
        noverlap=fs // 2,
        axis=-1,
    )
    frequency_step = frequencies[1] - frequencies[0]
    powers = []
    for low, high in BANDS:
        upper_edge = frequencies <= high if high == BANDS[-1][1] else frequencies < high
        mask = (frequencies >= low) & upper_edge
        powers.append(density[..., mask].sum(axis=-1) * frequency_step)
    log_power = np.log(np.maximum(np.stack(powers, axis=-1), 1e-12))
    return log_power.reshape(eeg.shape[0], -1)


def _labels_for_trials(labels, n_trials, n_classes):
    labels = np.asarray(labels)
    if (
        labels.shape != (n_trials,)
        or labels.dtype.kind not in "biuf"
        or not np.isin(labels, np.arange(n_classes)).all()
    ):
        raise ValueError(f"labels must contain one integer from 0 to {n_classes - 1} per trial")
    return labels.astype(int)


def _require_all_labels(labels, partition, n_classes):
    if np.unique(labels).size != n_classes:
        raise ValueError(f"{partition} must contain trials for all {n_classes} classes")


def _pipeline(components=None):
    steps = [("scale", StandardScaler())]
    if components is not None:
        steps.append(("pca", PCA(n_components=components, svd_solver="full")))
    steps.append(("classifier", LogisticRegression()))
    return Pipeline(steps)


def _check_pca(components, features):
    components = _integer(components, "components", 1)
    maximum = min(features.shape[0] - 1, features.shape[1])
    if components > maximum:
        raise ValueError(f"components must be at most {maximum} with these training trials")
    if not np.any(np.ptp(features, axis=0)):
        raise ValueError("posterior spectral features are constant across training trials")
    return components


def _posterior_features(features):
    return features[:, : POSTERIOR_CHANNEL_COUNT * len(BANDS)]


def _subtract_baseline(features, baseline_features):
    if baseline_features.shape != features.shape:
        raise ValueError("Imagery and baseline EEG must have the same trial and channel counts")
    return features - baseline_features


def decoder_features(eeg, fs, *, baseline_eeg=None):
    """Build posterior log-power features for training or predicting.

    If supplied, subtract log power of a neutral EEG window from the same
    trial and channel. The two windows can have different sample counts.
    """
    features = spectral_features(eeg, fs)
    if baseline_eeg is not None:
        features = _subtract_baseline(features, spectral_features(baseline_eeg, fs))
    return _posterior_features(features)


def fit_decoder(eeg, labels, fs, *, components=5, n_classes=2, baseline_eeg=None):
    """Fit the decoder on training data supplied by the caller.

    Set components to None to skip PCA. Use decoder_features with the same
    baseline convention to transform new EEG before predicting.
    """
    features = decoder_features(eeg, fs, baseline_eeg=baseline_eeg)
    n_classes = _integer(n_classes, "n_classes", 2)
    labels = _labels_for_trials(labels, len(features), n_classes)
    _require_all_labels(labels, "training data", n_classes)
    if components is not None:
        components = _check_pca(components, features)
    return _pipeline(components).fit(features, labels)


def _permuted_labels(labels, blocks, permutations, seed):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(blocks == block) for block in np.unique(blocks)]
    shuffled = np.tile(labels, (permutations, 1))
    for permutation in shuffled:
        for group in groups:
            permutation[group] = rng.permutation(labels[group])
    return shuffled


def _score(predictions, labels, n_classes):
    correct = predictions == labels
    matrix = confusion_matrix(labels, predictions, labels=np.arange(n_classes))
    per_class = []
    for label, row in enumerate(matrix):
        total = int(row.sum())
        hits = int(row[label])
        per_class.append({"label": label, "correct": hits, "total": total, "accuracy": hits / total})
    return {
        "accuracy": float(correct.mean()),
        "correct": int(correct.sum()),
        "total": len(labels),
        "predictions": predictions.tolist(),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


def _evaluate(features, labels, train, test, n_classes, components=None, permutations=()):
    train_x, test_x = features[train], features[test]
    train_y, test_y = labels[train], labels[test]
    model = _pipeline(components).fit(train_x, train_y)
    result = _score(model.predict(test_x), test_y, n_classes)
    result["shuffled_mean_accuracy"] = None
    result["permutation_p"] = None
    if len(permutations):
        # Scaling and PCA use features alone, so the same train-only transforms
        # can be reused when shuffling labels for the logistic regression fits.
        transform = model[:-1]
        train_x, test_x = transform.transform(train_x), transform.transform(test_x)
        scores = [
            LogisticRegression().fit(train_x, shuffled).score(test_x, test_y)
            for shuffled in permutations
        ]
        result["shuffled_mean_accuracy"] = float(np.mean(scores))
        result["permutation_p"] = (1 + sum(s >= result["accuracy"] for s in scores)) / (
            1 + len(scores)
        )
    return result


def compare(
    eeg, labels, blocks, fs, *, holdout_blocks=2, components=5, permutations=199, seed=7,
    n_classes=2, baseline_eeg=None,
):
    """Compare the same logistic classifier with and without PCA on later blocks.

    Permutations shuffle training labels within each training block, refit the
    classifier, and score it against the original held-out labels. Reported
    permutation p-values compare each pipeline against its own shuffled null.

    If paired neutral EEG is supplied, also evaluate the classifier on the
    difference between imagery and neutral log power with the same split.
    """
    features = spectral_features(eeg, fs)
    n_classes = _integer(n_classes, "n_classes", 2)
    labels = _labels_for_trials(labels, len(features), n_classes)
    blocks = np.asarray(blocks)
    if (
        blocks.shape != labels.shape
        or blocks.dtype.kind not in "iu"
        or np.any(blocks < 1)
        or np.any(blocks[1:] < blocks[:-1])
    ):
        raise ValueError("blocks must contain positive chronological integers, one per trial")

    holdout_blocks = _integer(holdout_blocks, "holdout_blocks", 1)
    permutations = _integer(permutations, "permutations", 0)
    unique_blocks = np.unique(blocks)
    if len(unique_blocks) <= holdout_blocks:
        raise ValueError(f"need at least {holdout_blocks + 1} blocks to leave training data")
    train_blocks, test_blocks = unique_blocks[:-holdout_blocks], unique_blocks[-holdout_blocks:]
    train, test = blocks < test_blocks[0], blocks >= test_blocks[0]
    _require_all_labels(labels[train], "training data", n_classes)
    _require_all_labels(labels[test], "held-out data", n_classes)

    posterior = _posterior_features(features)
    rest = None if baseline_eeg is None else spectral_features(baseline_eeg, fs)
    components = _check_pca(components, posterior[train])
    shuffled = _permuted_labels(labels[train], blocks[train], permutations, seed)
    baseline = _evaluate(posterior, labels, train, test, n_classes, permutations=shuffled)
    svd = _evaluate(posterior, labels, train, test, n_classes, components, shuffled)
    truth = labels[test]
    baseline_correct = np.asarray(baseline["predictions"]) == truth
    svd_correct = np.asarray(svd["predictions"]) == truth

    per_block = []
    for block in test_blocks:
        mask = blocks[test] == block
        per_block.append({
            "block": int(block),
            "total": int(mask.sum()),
            "baseline_correct": int(baseline_correct[mask].sum()),
            "svd_correct": int(svd_correct[mask].sum()),
            "baseline_accuracy": float(baseline_correct[mask].mean()),
            "svd_accuracy": float(svd_correct[mask].mean()),
        })

    result = {
        "train_blocks": train_blocks.tolist(),
        "test_blocks": test_blocks.tolist(),
        "test_indices": np.flatnonzero(test).tolist(),
        "labels": truth.tolist(),
        "n_classes": n_classes,
        "chance_accuracy": 1 / n_classes,
        "components": components,
        "permutations": permutations,
        "baseline": baseline,
        "svd": svd,
        "svd_accuracy_delta": svd["accuracy"] - baseline["accuracy"],
        "svd_only_correct": int((svd_correct & ~baseline_correct).sum()),
        "baseline_only_correct": int((baseline_correct & ~svd_correct).sum()),
        "per_block": per_block,
    }
    if rest is not None:
        normalized = _posterior_features(_subtract_baseline(features, rest))
        paired = _evaluate(normalized, labels, train, test, n_classes, permutations=shuffled)
        result["baseline_normalized"] = paired
        result["baseline_normalized_accuracy_delta"] = paired["accuracy"] - baseline["accuracy"]
        result["rest_control"] = _evaluate(_posterior_features(rest), labels, train, test,
                                           n_classes, permutations=shuffled)
    if features.shape[1] == 8 * len(BANDS):
        frontal = features[:, POSTERIOR_CHANNEL_COUNT * len(BANDS):]
        result["frontal_control"] = _evaluate(frontal, labels, train, test, n_classes)
    return result
