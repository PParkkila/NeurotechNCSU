"""Frequency-tagged gaze selection from NeuroPawn EEG."""

import numpy as np
from scipy.signal import butter, sosfiltfilt


def rail_mask(eeg):
    """Flag the observed ADC endpoints for the standard Knight in BrainFlow 5.23."""
    eeg = np.asarray(eeg)
    return (eeg >= 333333) | (eeg <= -333343)


def _basis(samples):
    centered = samples - samples.mean(axis=0)
    vectors, singular, _ = np.linalg.svd(centered, full_matrices=False)
    tolerance = singular[0] * max(centered.shape) * np.finfo(float).eps
    return vectors[:, singular > tolerance]


def score_targets(eeg, fs, frequencies=(10.0, 12.0)):
    """Return the leading CCA correlation for each visible flicker frequency.

    ``eeg`` is channels by samples. Sine and cosine references at the first
    and second harmonics let the decoder match any stimulus phase.
    """
    eeg = np.asarray(eeg)
    frequencies = np.asarray(frequencies, dtype=float)
    if not np.isfinite(fs) or fs <= 70:
        raise ValueError("The SSVEP bandpass needs a sampling rate above 70 Hz.")
    if (
        frequencies.shape != (2,)
        or not np.isfinite(frequencies).all()
        or np.any((frequencies <= 6) | (2 * frequencies >= 35))
        or frequencies[0] == frequencies[1]
    ):
        raise ValueError("Choose two distinct flicker frequencies and keep their harmonics in the 6–35 Hz band.")
    if eeg.ndim != 2 or not eeg.shape[0] or eeg.shape[1] < fs:
        raise ValueError("Provide at least one second of EEG as channels by samples.")
    if eeg.dtype.kind not in "iuf" or not np.isfinite(eeg).all():
        raise ValueError("EEG must contain real, finite samples.")
    if np.any(np.ptp(eeg, axis=1) == 0):
        raise ValueError("A selected EEG channel is flat.")
    if np.any(rail_mask(eeg)):
        raise ValueError("A selected EEG channel is clipped at the ADC limit.")

    bandpass = butter(4, (6, 35), btype="bandpass", fs=fs, output="sos")
    eeg_basis = _basis(sosfiltfilt(bandpass, eeg, axis=-1).T)
    if not eeg_basis.shape[1]:
        raise ValueError("The selected EEG channels have no variation after filtering.")

    seconds = np.arange(eeg.shape[1]) / fs
    scores = []
    for frequency in frequencies:
        angles = 2 * np.pi * frequency * seconds[:, None] * np.array([1, 2])
        reference = _basis(np.column_stack((np.sin(angles), np.cos(angles))))
        singular_values = np.linalg.svd(eeg_basis.T @ reference, compute_uv=False)
        scores.append(singular_values[0])
    return np.clip(scores, 0, 1)


def _scores(values):
    scores = np.asarray(values, dtype=float)
    if (
        not scores.ndim
        or scores.shape[-1] != 2
        or not np.isfinite(scores).all()
        or np.any((scores < 0) | (scores > 1))
    ):
        raise ValueError("Provide two finite CCA scores between zero and one per trial.")
    return scores


def fit_gate(scores, labels):
    """Fit a conservative decision gate to cued and idle calibration trials.

    Labels 0 and 1 identify the cued target, and -1 identifies idle. An
    accepted result must exceed both the largest idle score and the largest
    idle margin. A wrong cued result that exceeds the score cutoff can raise
    the margin cutoff further.
    """
    scores = _scores(scores)
    labels = np.asarray(labels)
    if (
        scores.ndim != 2
        or labels.shape != (len(scores),)
        or not np.array_equal(np.unique(labels), [-1, 0, 1])
    ):
        raise ValueError("Calibration requires labeled idle trials and trials for both targets.")

    peak = scores.max(axis=1)
    difference = np.abs(scores[:, 0] - scores[:, 1])
    winner = scores.argmax(axis=1)
    idle = labels == -1
    threshold = float(peak[idle].max())
    wrong = (labels != -1) & (winner != labels) & (peak > threshold)
    margin = float(difference[idle | wrong].max())
    accepted = (peak > threshold) & (difference > margin)
    if not all(np.any(accepted & (labels == target)) for target in (0, 1)):
        raise ValueError("Calibration could not separate both targets from idle and wrong selections.")
    return {"threshold": threshold, "margin": margin}


def select(scores, gate):
    """Return the selected target index, or None when the gate abstains."""
    scores = _scores(scores)
    if scores.ndim != 1:
        raise ValueError("Provide the two CCA scores from one trial.")
    if scores.max() > gate["threshold"] and abs(scores[0] - scores[1]) > gate["margin"]:
        return int(scores.argmax())
    return None
