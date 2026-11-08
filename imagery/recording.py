"""Acquire visual imagery trials from a NeuroPawn Knight Board."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from numbers import Integral
from shutil import get_terminal_size
from typing import NotRequired, TypedDict

import numpy as np


class Session(TypedDict):
    eeg: np.ndarray
    labels: np.ndarray
    blocks: np.ndarray
    fs: int
    baseline: NotRequired[np.ndarray]


class ADCRailError(RuntimeError):
    """A saturated input prevented acquisition of a usable capture."""


def adc_rail_mask(eeg: np.ndarray) -> np.ndarray:
    """Flag the observed standard Knight ADC endpoints in BrainFlow 5.23."""
    eeg = np.asarray(eeg)
    return (eeg >= 333333) | (eeg <= -333343)


class NeuroPawn:
    """Stream selected EEG inputs from one standard or IMU board."""

    def __init__(
        self,
        port: str,
        *,
        imu: bool = False,
        channels: int = 8,
        gain: int = 12,
        inputs: Sequence[int] | None = None,
    ):
        if not port:
            raise ValueError("A NeuroPawn USB serial port is required.")
        if (
            isinstance(channels, bool)
            or not isinstance(channels, Integral)
            or not 1 <= channels <= 8
        ):
            raise ValueError("The NeuroPawn supports between 1 and 8 EEG channels.")
        if isinstance(gain, bool) or not isinstance(gain, Integral) or gain not in (1, 2, 3, 4, 6, 8, 12):
            raise ValueError("NeuroPawn gain must be one of 1, 2, 3, 4, 6, 8, or 12.")
        if inputs is None:
            selected = tuple(range(1, channels + 1))
        else:
            if channels != 8:
                raise ValueError("Specify either an EEG channel count or explicit NeuroPawn inputs.")
            selected = tuple(inputs)
            if (
                not selected
                or any(
                    isinstance(channel, bool)
                    or not isinstance(channel, Integral)
                    or not 1 <= channel <= 8
                    for channel in selected
                )
                or len(set(selected)) != len(selected)
            ):
                raise ValueError("NeuroPawn inputs must be unique integers between 1 and 8.")

        self.port = port
        self.imu = imu
        self.inputs = selected
        self.channels = len(selected)
        self.gain = int(gain)
        self._explicit_inputs = inputs is not None
        self.fs: int
        self._board = None
        self._marker = 0

    def __enter__(self) -> NeuroPawn:
        if self._board is not None:
            raise RuntimeError("This NeuroPawn session is already open.")
        try:
            from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
            from brainflow.exit_codes import BrainFlowError
        except ImportError as exc:
            raise RuntimeError("Install BrainFlow to record from a NeuroPawn.") from exc

        board_name = "NEUROPAWN_KNIGHT_BOARD_IMU" if self.imu else "NEUROPAWN_KNIGHT_BOARD"
        board_kind = getattr(BoardIds, board_name, None)
        if board_kind is None:
            raise RuntimeError(f"The installed BrainFlow version does not support {board_name}.")
        board_id = board_kind.value
        self.fs = BoardShim.get_sampling_rate(board_id)
        eeg_rows = BoardShim.get_eeg_channels(board_id)
        self._eeg_rows = [eeg_rows[channel - 1] for channel in self.inputs]
        self._marker_row = BoardShim.get_marker_channel(board_id)
        self._packet_row = BoardShim.get_package_num_channel(board_id)

        params = BrainFlowInputParams()
        params.serial_port = self.port
        params.other_info = json.dumps({"gain": self.gain})
        board = BoardShim(board_id, params)
        try:
            board.prepare_session()
            board.start_stream()
            print("Configuring NeuroPawn channels.", file=sys.stderr)
            # The firmware can silently drop register writes sent too quickly.
            time.sleep(3.0)
            if self._explicit_inputs:
                for channel in sorted(set(range(1, 9)) - set(self.inputs)):
                    board.config_board(f"rldremove_{channel}")
                    time.sleep(1.5)
                    board.config_board(f"choff_{channel}")
                    time.sleep(1.5)
            for channel in self.inputs:
                board.config_board(f"chon_{channel}_{self.gain}")
                time.sleep(1.5)
                board.config_board(f"rldadd_{channel}")
                time.sleep(1.5)
        except BaseException as exc:
            with suppress(Exception):
                board.release_session()
            if isinstance(exc, BrainFlowError):
                raise RuntimeError(f"Could not start {board_name} on {self.port}. {exc}") from exc
            raise

        self._board = board
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        board, self._board = self._board, None
        if board is None:
            return
        # Releasing a Knight session also stops the stream.
        if exc_type is None:
            board.release_session()
        else:
            with suppress(Exception):
                board.release_session()

    def capture(
        self, seconds: float, *, on_start: Callable[[], None] | None = None
    ) -> np.ndarray:
        """Return fresh EEG samples starting at an unlabeled software marker."""
        board = self._board
        if board is None:
            raise RuntimeError("Open the NeuroPawn using a with block before capturing.")
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError("Capture duration must be a positive finite number of seconds.")
        sample_count = round(seconds * self.fs)
        if not sample_count:
            raise ValueError("Capture duration is shorter than one sample.")

        board.get_board_data()
        if on_start is not None:
            on_start()
        # A unique marker keeps a timed-out capture from matching a later trial.
        self._marker += 1
        board.insert_marker(float(self._marker))
        deadline = time.monotonic() + seconds + 3.0
        chunks = []
        collected = 0
        while time.monotonic() < deadline:
            data = board.get_board_data()
            if data.shape[1] == 0:
                time.sleep(0.02)
                continue
            if collected == 0:
                marker_positions = np.flatnonzero(data[self._marker_row] == self._marker)
                if marker_positions.size == 0:
                    continue
                data = data[:, marker_positions[0] :]
            chunks.append(data)
            collected += data.shape[1]
            if collected >= sample_count:
                samples = np.concatenate(chunks, axis=1)[:, :sample_count]
                packets = samples[self._packet_row].astype(np.int64)
                if np.any(np.diff(packets) % 256 != 1):
                    raise RuntimeError("A NeuroPawn packet was lost during capture. Check the USB connection.")
                eeg = samples[self._eeg_rows, :]
                flat = np.ptp(eeg, axis=1) == 0
                if not self.imu and np.any(flat) and np.any(adc_rail_mask(eeg[flat, 0])):
                    raise ADCRailError("A NeuroPawn EEG channel is flat or invalid. Check the electrodes.")
                if not np.isfinite(eeg).all() or np.any(flat):
                    raise RuntimeError("A NeuroPawn EEG channel is flat or invalid. Check the electrodes.")
                return eeg

        raise TimeoutError(
            f"NeuroPawn supplied {collected}/{sample_count} samples after the capture marker. "
            "Check the serial port, cable, and board variant."
        )


_SHAPE_PREVIEWS = (
    "\n".join((
        "     #######     ",
        "   ##       ##   ",
        "  #           #  ",
        " #             # ",
        "#               #",
        " #             # ",
        "  #           #  ",
        "   ##       ##   ",
        "     #######     ",
    )),
    "#" * 17 + "\n" + ("#" + " " * 15 + "#\n") * 7 + "#" * 17,
)

MOVE_NAMES = ("Circle", "Triangle", "Square")


def _show(cue: str, message: str = "") -> None:
    columns, rows = get_terminal_size()
    lines = cue.split("\n")[:rows]
    row = max(1, rows // 2 - len(lines) // 2)
    width = max(map(len, lines), default=0)
    column = max(1, (columns - width) // 2 + 1)
    rendered = "".join(
        f"\x1b[{row + offset};{column}H{line[:columns]}"
        for offset, line in enumerate(lines)
    )
    message_row = min(rows, row + len(lines) + 1)
    print(
        f"\x1b[2J{rendered}\x1b[{message_row};1H{message}",
        end="",
        flush=True,
    )


def capture_move(
    board: NeuroPawn, *, seconds: float, skip: float, baseline_seconds: float = 2.0,
    on_phase: Callable[[str], None] | None = None, before_go: Callable[[], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Capture paired rest and imagery, optionally giving a cue after rest."""
    if not np.isfinite([seconds, skip, baseline_seconds]).all() or skip < 0:
        raise ValueError("Use finite durations and a nonnegative imagery delay.")
    baseline_samples = round(baseline_seconds * board.fs)
    imagery_samples = round(seconds * board.fs)
    if min(baseline_samples, imagery_samples) < 1:
        raise ValueError("Rest and imagery durations must each span at least one sample.")
    duration = (imagery_samples + round(skip * board.fs)) / board.fs

    if on_phase is None:
        _show("+")
    else:
        on_phase("rest")
    time.sleep(0.5)  # Let typing or refixation after the previous round settle.
    baseline = board.capture(baseline_samples / board.fs)
    if np.any(adc_rail_mask(baseline)):
        raise ADCRailError("A NeuroPawn EEG channel clipped during rest.")
    if before_go is not None:
        before_go()
    go = (lambda: _show("+", "GO")) if on_phase is None else (lambda: on_phase("go"))
    imagery = board.capture(duration, on_start=go)
    if on_phase is None:
        _show("+")
    if np.any(adc_rail_mask(imagery)):
        raise ADCRailError("A NeuroPawn EEG channel clipped during imagery.")
    return baseline, imagery[:, -imagery_samples:]


def report_move() -> int:
    """Ask for the privately imagined shape after the EEG capture ends."""
    names = tuple(name.lower() for name in MOVE_NAMES)
    while True:
        answer = input("\nWhat did you imagine? Type circle, triangle, or square and press Enter. ")
        answer = answer.strip().casefold()
        if answer in names:
            return names.index(answer)


def _collect_self_selected(
    board: NeuroPawn,
    *,
    blocks: int,
    trials: int,
    seconds: float,
    skip: float,
    on_block: Callable[[Session], None] | None,
) -> Session:
    if blocks < 1 or trials < len(MOVE_NAMES) or trials % len(MOVE_NAMES):
        raise ValueError("Choose at least one block and a positive multiple of three trials per block.")
    quota = trials // len(MOVE_NAMES)
    recordings = []
    baselines = []
    labels = []
    block_numbers = []

    for block in range(1, blocks + 1):
        _show("+", f"Block {block}/{blocks}. Choose a shape privately at GO. Use each shape {quota} times in mixed order.")
        input("\nDuring rest, look at + and wait for GO to start imagining. Press Enter to begin.")
        counts = [0] * len(MOVE_NAMES)
        clipped = 0
        over_quota = 0
        while sum(counts) < trials:
            try:
                baseline, imagery = capture_move(board, seconds=seconds, skip=skip)
            except ADCRailError as exc:
                clipped += 1
                if clipped == 3:
                    raise RuntimeError("Three captures clipped in a row. Check the electrodes and ear clip.") from exc
                _show("+", "EEG clipped. Reseat the electrodes or ear clip and settle before retrying.")
                input("\nPress Enter to start a fresh round.")
                continue
            clipped = 0

            move = report_move()
            if counts[move] == quota:
                over_quota += 1
                if over_quota == 6:
                    raise RuntimeError("Six reported moves exceeded the block quota. Use each shape equally when restarting.")
                _show("+", "That shape has already filled its quota for this block.")
                input("\nPress Enter to start a fresh round.")
                continue
            over_quota = 0
            counts[move] += 1
            baselines.append(baseline)
            recordings.append(imagery)
            labels.append(move)
            block_numbers.append(block)

        session = Session(
            eeg=np.stack(recordings),
            baseline=np.stack(baselines),
            labels=np.asarray(labels),
            blocks=np.asarray(block_numbers),
            fs=board.fs,
        )
        if on_block is not None:
            on_block(session)

    _show("+", "Recording complete.\n")
    return session


def collect(
    board: NeuroPawn,
    *,
    blocks: int = 5,
    trials: int = 20,
    seconds: float = 3.0,
    seed: int | None = None,
    cue_seconds: float = 1.0,
    washout: float = 2.0,
    skip: float = 0.75,
    task: str = "layout",
    on_block: Callable[[Session], None] | None = None,
) -> Session:
    """Record balanced blocks with private or cued visual imagery."""
    if task == "rps":
        return _collect_self_selected(
            board, blocks=blocks, trials=trials, seconds=seconds, skip=skip, on_block=on_block
        )
    if blocks < 1 or trials < 2 or trials % 2:
        raise ValueError("Choose at least one block and an even number of trials per block.")
    if task == "shape_preview":
        cues = _SHAPE_PREVIEWS
        legend = "Look at the shape. Picture it in the center after it disappears."
    elif task == "shape":
        cues = ("CIRCLE", "SQUARE")
        legend = "Imagine one shape in the center."
    elif task == "layout":
        cues = ("A", "B")
        legend = "A = circle above square. B = square above circle."
    else:
        raise ValueError("Choose the layout, shape, shape_preview, or rps task.")
    timing = np.array([seconds, cue_seconds, washout, skip])
    if not np.isfinite(timing).all() or np.any(timing < 0) or min(seconds, cue_seconds) <= 0:
        raise ValueError("Imagery and cue durations must be positive, and delays cannot be negative.")

    kept_samples = round(seconds * board.fs)
    if kept_samples < 1:
        raise ValueError("Imagery duration is shorter than one sample.")
    duration = (kept_samples + round(skip * board.fs)) / board.fs
    random = np.random.default_rng(seed)
    labels_per_block = np.repeat([0, 1], trials // 2)
    recordings = []
    labels = []
    block_numbers = []

    for block in range(1, blocks + 1):
        _show("+", f"Block {block}/{blocks}. {legend}")
        input("\nLook at the cue, then fixate on + and start imagining at GO. Press Enter to begin.")
        for target in random.permutation(labels_per_block):
            _show(cues[target])
            time.sleep(cue_seconds)
            _show("+")
            time.sleep(washout)
            window = board.capture(duration, on_start=lambda: _show("+", "GO"))
            recordings.append(window[:, -kept_samples:])
            labels.append(target)
            block_numbers.append(block)

        session = Session(
            eeg=np.stack(recordings),
            labels=np.asarray(labels),
            blocks=np.asarray(block_numbers),
            fs=board.fs,
        )
        if on_block is not None:
            on_block(session)

    _show("+", "Recording complete.\n")
    return session
