"""Record, evaluate, and draw predictions from a NeuroPawn session."""

import argparse
from functools import partial
import json
from pathlib import Path
import sys
import tempfile
import time
import webbrowser


def save_session(path, data, *, source, skip):
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    stream = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            np.savez_compressed(stream, **data, source=source, skip=skip)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_session(path):
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        task = str(archive["task"].item()) if "task" in archive else "layout"
        if task not in ("layout", "shape", "shape_preview", "rps"):
            raise ValueError(f"Unknown imagery task {task!r} in {path}.")
        return {
            "eeg": archive["eeg"],
            "labels": archive["labels"],
            "blocks": archive["blocks"],
            "fs": archive["fs"].item(),
            "source": str(archive["source"].item()),
            "skip": float(archive["skip"].item()),
            "task": task,
            "gain": int(archive["gain"].item()) if "gain" in archive else None,
            "inputs": tuple(archive["inputs"].tolist()) if "inputs" in archive else None,
            "baseline": archive["baseline"] if "baseline" in archive else None,
        }


def drawing_for(task):
    from .drawing import LAYOUT_NAMES, SHAPE_NAMES, draw_layout, draw_shape

    if task == "rps":
        from .recording import MOVE_NAMES

        return partial(draw_shape, names=MOVE_NAMES), MOVE_NAMES
    return (draw_layout, LAYOUT_NAMES) if task == "layout" else (draw_shape, SHAPE_NAMES)


def record(args):
    from .recording import NeuroPawn, collect

    if args.out.exists():
        raise ValueError(f"Choose a new path; {args.out} already exists.")
    task = args.task.replace("-", "_")
    is_rps = task == "rps"
    trials = args.trials or (12 if is_rps else 20)
    class_count = 3 if is_rps else 2
    if trials % class_count:
        raise ValueError(f"Use a multiple of {class_count} trials per block.")
    if is_rps and args.blocks < 3:
        raise ValueError("Record at least three blocks for the default train and holdout split.")
    skip = 1.0 if is_rps else 0.75
    gain = args.gain or (4 if is_rps else 12)
    inputs = None
    if is_rps:
        inputs = tuple(range(1, args.channels + 1)) if args.channels else (1, 2, 3, 6)

    def checkpoint(data):
        save_session(
            args.out, dict(data, task=task, gain=gain, inputs=board.inputs),
            source="neuropawn", skip=skip,
        )

    if is_rps:
        print("Choose circle, triangle, or square in your mind at GO. Report it after capture.")
        print(f"Using board inputs {inputs} at gain {gain}.")
    elif task == "shape_preview":
        print("Look at the preview. At GO, picture the shape that just disappeared.")
    elif task == "shape":
        print("Read CIRCLE or SQUARE and imagine that shape in the center.")
    else:
        print("A = circle above square. B = square above circle.")
    print("Fixate on + and keep your gaze, hands, and jaw still.")
    print("The recording is saved locally after each complete block.", flush=True)
    try:
        with NeuroPawn(args.port, imu=args.imu, channels=8 if inputs else args.channels or 8,
                       gain=gain, inputs=inputs) as board:
            data = collect(
                board, blocks=args.blocks, trials=trials, seed=args.seed,
                seconds=4.0 if is_rps else 3.0, skip=skip, task=task, on_block=checkpoint,
            )
    except KeyboardInterrupt:
        if args.out.exists():
            print(f"\nCompleted blocks saved in {args.out}.", file=sys.stderr)
        raise
    print(f"Saved {len(data['labels'])} trials to {args.out}.")


def simulate(args):
    import numpy as np

    if args.out.exists():
        raise ValueError(f"Choose a new path; {args.out} already exists.")
    rng = np.random.default_rng(args.seed)
    fs = 125
    if args.task == "rps":
        blocks = np.repeat(np.arange(1, 6), 12)
        labels = np.concatenate([rng.permutation(np.repeat(np.arange(3), 4)) for _ in range(5)])
        baseline = rng.normal(0, 6, (len(labels), 4, 2 * fs))
        eeg = rng.normal(0, 6, (len(labels), 4, 4 * fs))
        t = np.arange(4 * fs) / fs
        tones = 8 * np.sin(2 * np.pi * np.array([6, 10, 17])[labels, None] * t)
        eeg[:, :2] += tones[:, None, :]
        drift = rng.lognormal(0, 0.6, (len(labels), 4, 1))
        save_session(
            args.out, dict(eeg=eeg * drift, baseline=baseline * drift, labels=labels,
                           blocks=blocks, fs=fs, task="rps", inputs=(1, 2, 3, 6)),
            source="synthetic", skip=1.0,
        )
        print(f"Saved synthetic three-move test data to {args.out}. It contains no measured EEG.")
        return
    blocks = np.repeat(np.arange(1, 6), 20)
    labels = np.concatenate([rng.permutation(np.repeat([0, 1], 10)) for _ in range(5)])
    t = np.arange(3 * fs) / fs
    eeg = rng.normal(0, 6, (len(labels), 8, len(t)))

    # Exercise the pipeline with known signals and unrelated block variation.
    for trial, label in enumerate(labels):
        alpha = rng.uniform(4, 15) * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi))
        tone = rng.uniform(1, 6) * np.sin(2 * np.pi * (6 if label == 0 else 17) * t)
        eeg[trial, :6] += alpha
        eeg[trial, :2] += tone
        eeg[trial, :6] *= 1 + 0.1 * blocks[trial]

    save_session(args.out, dict(eeg=eeg, labels=labels, blocks=blocks, fs=fs),
                 source="synthetic", skip=0.75)
    print(f"Saved synthetic test data to {args.out}. It contains no measured EEG.")


def evaluate(args):
    from .model import compare

    if len({path.resolve() for path in (args.session, args.out, args.svg)}) != 3:
        raise ValueError("Use different paths for the session, JSON report, and SVG drawing.")
    data = load_session(args.session)
    draw, names = drawing_for(data["task"])
    if data["task"] == "rps" and data["baseline"] is None:
        raise ValueError("The three-move session must contain paired resting EEG.")
    report = compare(
        data["eeg"], data["labels"], data["blocks"], data["fs"],
        holdout_blocks=args.holdout_blocks, components=args.components,
        permutations=args.permutations, seed=args.seed, n_classes=len(names),
        baseline_eeg=data["baseline"],
    )
    report["source"] = data["source"]
    report["task"] = data["task"]
    report["class_names"] = names
    report["gain"] = data["gain"]
    report["inputs"] = data["inputs"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{data['source'].title()} EEG · Held-out blocks {report['test_blocks']} · "
          f"Uniform-guess accuracy {report['chance_accuracy']:.1%}")
    for title, key in (("Fourier only", "baseline"), ("Fourier + SVD", "svd"),
                       ("Fourier minus resting baseline", "baseline_normalized"),
                       ("Pre-GO rest only", "rest_control"),
                       ("Frontal control", "frontal_control")):
        result = report.get(key)
        if result is not None:
            print(f"{title}: {result['correct']}/{result['total']} ({result['accuracy']:.1%})")
            if result.get("permutation_p") is not None:
                print(f"  Shuffled-label p = {result['permutation_p']:.3f}")
    print(f"SVD accuracy difference: {100 * report['svd_accuracy_delta']:+.1f} percentage points")
    if "baseline_normalized_accuracy_delta" in report:
        print(f"Rest subtraction difference: "
              f"{100 * report['baseline_normalized_accuracy_delta']:+.1f} percentage points")
    print(f"Only SVD correct: {report['svd_only_correct']}; "
          f"only Fourier correct: {report['baseline_only_correct']}")

    decoder = report.get("baseline_normalized", report["svd"])
    # Draw the first held-out prediction, regardless of correctness.
    path = draw(args.svg, decoder["predictions"][0], actual=report["labels"][0],
                trial=report["test_indices"][0] + 1,
                subtitle=f"{data['source'].title()} · First held-out trial")
    print(f"Report: {args.out}\nDrawing: {path}")
    if args.open:
        webbrowser.open(path.resolve().as_uri())


def live(args):
    import numpy as np

    from .model import POSTERIOR_CHANNEL_COUNT, fit_decoder, spectral_features
    from .recording import NeuroPawn

    if args.session.resolve() == args.out.resolve():
        raise ValueError("Use different paths for the session and SVG drawing.")
    data = load_session(args.session)
    if data["task"] == "rps":
        raise ValueError("Use play for a circle, triangle, and square session.")
    draw, names = drawing_for(data["task"])
    if data["source"] != "neuropawn":
        raise ValueError("Live drawing requires a session recorded from a NeuroPawn.")
    if args.gain is not None and data["gain"] is not None and args.gain != data["gain"]:
        raise ValueError("Live drawing must use the gain from the recorded session.")
    gain = data["gain"] or args.gain or 12
    blocks = np.unique(data["blocks"])
    if len(blocks) <= args.holdout_blocks:
        raise ValueError("Record enough blocks to leave at least one training block.")
    train = np.isin(data["blocks"], blocks[:-args.holdout_blocks])
    model = fit_decoder(data["eeg"][train], data["labels"][train], data["fs"],
                        components=args.components)
    channels, samples = data["eeg"].shape[1:]
    print("Use the same wearer, electrodes, and channel order as the recorded session.")
    print(f"Choose between {names[0]} and {names[1]}.")
    with NeuroPawn(args.port, imu=args.imu, channels=channels, gain=gain) as board:
        if board.fs != data["fs"]:
            raise ValueError("The headset sample rate differs from the recorded session.")
        capture_samples = round(data["skip"] * board.fs) + samples
        for trial in range(1, args.trials + 1):
            input("\nChoose privately. Press Enter, then fixate on +. ")
            print("+", flush=True)
            time.sleep(2)
            epoch = board.capture(capture_samples / board.fs,
                                  on_start=lambda: print("GO", flush=True))
            features = spectral_features(epoch[None, :POSTERIOR_CHANNEL_COUNT, -samples:], board.fs)
            probabilities = model.predict_proba(features)[0]
            choice = int(np.argmax(probabilities))
            prediction = int(model.classes_[choice])
            score = float(probabilities[choice])
            path = draw(args.out, prediction, confidence=score, trial=trial,
                        subtitle="Live NeuroPawn · Experimental prediction")
            print(f"Prediction: {names[prediction]} (model score {score:.0%})")
            print(f"Drawing: {path}")
    if args.open:
        webbrowser.open(path.resolve().as_uri())


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Use a positive integer.")
    return result


def select_shapes(args):
    from .selection import run_selection

    run_selection(args)


def play(args):
    from .rps import play_game

    if args.profile is not None and args.profile.suffix != ".sqlite3":
        raise ValueError("Use a .sqlite3 filename for the local player profile.")
    if args.out is None:
        if args.profile is None:
            args.out = Path("runs/rps-game.jsonl")
        else:
            from uuid import uuid4

            args.out = Path("runs") / f"rps-teaching-{uuid4().hex}.jsonl"
    play_game(args, load_session(args.session))


def ordered_inputs(value):
    try:
        inputs = tuple(int(channel) for channel in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Enter comma-separated board input numbers.") from exc
    if any(channel < 1 or channel > 8 for channel in inputs) or len(inputs) != len(set(inputs)):
        raise argparse.ArgumentTypeError("Use distinct board inputs from 1 to 8.")
    return inputs


def web(args):
    from .web import run_web

    if args.listen_port > 65535:
        raise ValueError("Choose a listen port between 1 and 65535.")
    run_web(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    capture = commands.add_parser("record", help="Collect labeled trials from one NeuroPawn")
    capture.add_argument("--out", type=Path, default=Path("runs/session.npz"))
    capture.add_argument("--blocks", type=positive_int, default=5)
    capture.add_argument("--trials", type=positive_int, help="Trials per block (12 for rps, 20 otherwise)")
    capture.add_argument("--channels", type=int, choices=(6, 8), help="Override the default board inputs")
    capture.add_argument("--task", choices=("layout", "shape", "shape-preview", "rps"), default="layout",
                         help="Use layouts, cued shapes, or private three-move imagery")
    capture.add_argument("--seed", type=int)
    capture.set_defaults(run=record)

    mock = commands.add_parser("simulate", help="Create synthetic data to check the software")
    mock.add_argument("--out", type=Path, default=Path("runs/synthetic.npz"))
    mock.add_argument("--seed", type=int, default=0)
    mock.add_argument("--task", choices=("layout", "rps"), default="layout")
    mock.set_defaults(run=simulate)

    score = commands.add_parser("evaluate", help="Compare Fourier and Fourier + SVD on held-out blocks")
    score.add_argument("--out", type=Path, default=Path("runs/report.json"))
    score.add_argument("--svg", type=Path, default=Path("runs/held-out.svg"))
    score.add_argument("--seed", type=int, default=0)
    score.add_argument("--permutations", type=int, default=199)
    score.set_defaults(run=evaluate)

    predict = commands.add_parser("live", help="Draw a new prediction from live EEG")
    predict.add_argument("--out", type=Path, default=Path("runs/live.svg"))
    predict.add_argument("--trials", type=positive_int, default=1)
    predict.set_defaults(run=live)

    game = commands.add_parser("play", help="Play a private imagined-shape game with an EEG decoder")
    game.add_argument("--port", required=True, help="USB serial port for the NeuroPawn")
    game.add_argument("--session", type=Path, required=True, help="Recorded three-move session")
    game.add_argument("--out", type=Path, help="New local round log (automatically named with --profile)")
    game.add_argument("--profile", type=Path, help="Teach and reuse a local player profile (.sqlite3)")
    game.add_argument("--holdout-blocks", type=positive_int, default=2,
                      help="Blocks to exclude from training for fixed-decoder play")
    game.add_argument("--rounds", type=positive_int, default=12)
    game.add_argument("--imu", action="store_true", help="Use the Knight Board IMU variant")
    game.set_defaults(run=play)

    bridge = commands.add_parser("web", help="Serve Shape Battle with per-session EEG calibration")
    bridge.add_argument("--port", required=True, help="USB serial port for the NeuroPawn")
    bridge.add_argument("--imu", action="store_true", help="Use the Knight Board IMU variant")
    bridge.add_argument("--inputs", type=ordered_inputs, default=(1, 2, 6),
                        help="Active board inputs in acquisition order (default 1,2,6)")
    bridge.add_argument("--gain", type=int, choices=(1, 2, 3, 4, 6, 8, 12), default=4)
    bridge.add_argument("--listen-port", type=positive_int, default=8765, help="Local browser port")
    bridge.add_argument("--sessions", type=Path, default=Path("runs/web-sessions"),
                        help="Local directory for per-session EEG files")
    bridge.set_defaults(run=web)

    selection = commands.add_parser("select", help="Validate and draw with flicker-based EEG selection")
    selection.add_argument("--port", required=True, help="USB serial port for the NeuroPawn")
    selection.add_argument("--gain", type=int, choices=(1, 2, 3, 4, 6, 8, 12), default=4)
    selection.add_argument("--out", type=Path, default=Path("runs"), help="Validation report directory")
    selection.add_argument("--open", action="store_true", help="Open the local browser app")
    selection.set_defaults(run=select_shapes)

    for command in (capture, predict):
        command.add_argument("--port", required=True, help="USB serial port for the NeuroPawn")
        command.add_argument("--imu", action="store_true", help="Use the Knight Board IMU variant")
        command.add_argument("--gain", type=int, choices=(1, 2, 3, 4, 6, 8, 12),
                             help="Board gain (rps defaults to 4; other tasks to 12; live uses recorded gain)")
    for command in (score, predict):
        command.add_argument("--session", type=Path, default=Path("runs/session.npz"))
        command.add_argument("--holdout-blocks", type=positive_int, default=2)
        command.add_argument("--components", type=positive_int, default=5)
        command.add_argument("--open", action="store_true", help="Open the SVG in your browser")

    args = parser.parse_args()
    try:
        args.run(args)
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130
    except (EOFError, ImportError, KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
