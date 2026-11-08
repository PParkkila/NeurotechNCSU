"""Collect and export the rapid-fire three-shape training dataset."""

import json
import time

import numpy as np

from .recording import ADCRailError, MOVE_NAMES, Session, _show, capture_move


BLOCK_COUNT = 10
PER_SHAPE = 5
TRIALS_PER_BLOCK = PER_SHAPE * len(MOVE_NAMES)
REST_SECONDS = 2
IMAGERY_SECONDS = 4
SKIP_SECONDS = 1
PROTOCOL = "rapid-fire-rps"


def collect_rapid(board, *, on_block, previous=None):
    """Play automatically cued, balanced blocks and checkpoint each finished block."""
    rng = np.random.default_rng()
    targets = np.repeat(np.arange(len(MOVE_NAMES)), PER_SHAPE)
    recordings = list(previous["eeg"]) if previous else []
    baselines = list(previous["baseline"]) if previous else []
    labels = list(previous["labels"]) if previous else []
    blocks = list(previous["blocks"]) if previous else []
    first_block = len(labels) // TRIALS_PER_BLOCK + 1

    for block in range(first_block, BLOCK_COUNT + 1):
        _show("+", f"Block {block}/{BLOCK_COUNT} · {len(labels)}/{BLOCK_COUNT * TRIALS_PER_BLOCK} trials saved")
        input("\nRest at +, read the shape name, then imagine it at GO. Press Enter when ready for 15 trials.")
        for trial, target in enumerate(rng.permutation(targets), start=1):
            def cue():
                _show(MOVE_NAMES[target].upper(), f"Block {block}/{BLOCK_COUNT} · Trial {trial}/{TRIALS_PER_BLOCK}")
                time.sleep(1)

            for attempt in range(3):
                try:
                    baseline, imagery = capture_move(
                        board, seconds=IMAGERY_SECONDS, skip=SKIP_SECONDS,
                        baseline_seconds=REST_SECONDS, before_go=cue,
                    )
                    break
                except ADCRailError as exc:
                    if attempt == 2:
                        raise RuntimeError("Three captures clipped in a row. Check the electrodes and reference.") from exc
                    _show("+", "EEG clipped. Check the electrodes and reference.")
                    input("\nPress Enter to retry this trial.")
            baselines.append(baseline)
            recordings.append(imagery)
            labels.append(target)
            blocks.append(block)

        on_block(Session(eeg=np.stack(recordings), baseline=np.stack(baselines),
                         labels=np.asarray(labels), blocks=np.asarray(blocks), fs=board.fs))

    _show("+", "Recording complete.\n")
    return len(labels)


def dataset_configuration(data, *, complete):
    """Check compatibility with the browser trainer and return its board settings."""
    fs, inputs, gain = data["fs"], data["inputs"], data["gain"]
    if (data["task"] != "rps" or data["source"] != "neuropawn" or data["protocol"] != PROTOCOL
            or data["skip"] != SKIP_SECONDS or fs != 125 or not inputs
            or any(type(channel) is not int or not 1 <= channel <= 8 for channel in inputs)
            or len(set(inputs)) != len(inputs) or gain not in (1, 2, 3, 4, 6, 8, 12)):
        raise ValueError("Use a 125 Hz dataset saved by record-rps150 with the original board settings.")

    labels, blocks = data["labels"], data["blocks"]
    count = labels.size
    block_count, extra = divmod(count, TRIALS_PER_BLOCK)
    if extra or not 1 <= block_count <= BLOCK_COUNT or (complete and block_count != BLOCK_COUNT):
        raise ValueError(f"Expected {'all ten' if complete else 'one to ten'} complete 15-trial blocks.")
    targets = np.repeat(np.arange(len(MOVE_NAMES)), PER_SHAPE)
    if (labels.shape != (count,) or labels.dtype.kind not in "iu"
            or not np.array_equal(blocks, np.repeat(np.arange(1, block_count + 1), TRIALS_PER_BLOCK))
            or not np.all(np.sort(labels.reshape(block_count, TRIALS_PER_BLOCK), axis=1) == targets)):
        raise ValueError("Each chronological block must have five trials of each shape.")
    for field, seconds in (("baseline", REST_SECONDS), ("eeg", IMAGERY_SECONDS)):
        samples = data[field]
        if (samples is None or samples.shape != (count, len(inputs), seconds * fs)
                or samples.dtype.kind not in "iuf" or not np.isfinite(samples).all()):
            raise ValueError(f"Invalid {field} samples for the recorded headset configuration.")
    return {"fs": fs, "inputs": inputs, "gain": gain, "imu": data["imu"]}


def export_dataset(path, data):
    configuration = dataset_configuration(data, complete=True)
    trials = [
        {"label": int(label), "block": int(block), "baseline": baseline.tolist(), "eeg": eeg.tolist()}
        for label, block, baseline, eeg in zip(data["labels"], data["blocks"], data["baseline"], data["eeg"], strict=True)
    ]
    payload = {"format": "neuropawn-rps-dataset", "version": 1, "configuration": configuration, "trials": trials}
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("x", encoding="utf-8")
    try:
        with stream:
            json.dump(payload, stream, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
