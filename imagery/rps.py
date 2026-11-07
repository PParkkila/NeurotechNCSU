"""Play the three-move imagery experiment against a random opponent."""

from contextlib import ExitStack
from hashlib import file_digest
import json
from secrets import randbelow
from uuid import uuid4

import numpy as np

from .model import decoder_features, fit_decoder
from .recording import ADCRailError, MOVE_NAMES, NeuroPawn, capture_move, report_move


def round_result(decoded, opponent):
    if decoded == opponent:
        return "Draw"
    return "Win" if (opponent - decoded) % len(MOVE_NAMES) == 1 else "Loss"


def _confirm_move(decoded):
    names = tuple(name.casefold() for name in MOVE_NAMES)
    while True:
        answer = input(
            f"Press Enter to confirm {MOVE_NAMES[decoded]}, type circle, triangle, or square to correct, "
            "or type skip. "
        ).strip().casefold()
        if not answer:
            return decoded
        if answer == "skip":
            return None
        if answer in names:
            return names.index(answer)


def play_game(args, data):
    if data["source"] != "neuropawn" or data["task"] != "rps" or data["baseline"] is None:
        raise ValueError("Play requires a real three-move recording with paired resting EEG.")
    if args.out.exists():
        raise ValueError(f"Choose a new round log. {args.out} already exists.")

    profile = getattr(args, "profile", None)
    eeg, labels, blocks = data["eeg"], data["labels"], data["blocks"]
    if not data["inputs"] or len(data["inputs"]) != eeg.shape[1] or data["gain"] is None:
        raise ValueError("The recording must identify the board inputs and gain used for training.")

    outcomes = {"Win": 0, "Loss": 0, "Draw": 0}
    counts = np.zeros(len(MOVE_NAMES), dtype=int)
    hits = np.zeros(len(MOVE_NAMES), dtype=int)
    seed_hits = 0
    candidate_hits = 0
    candidate_shown = 0
    clipped = 0
    unlabeled = 0
    with ExitStack() as stack:
        decoder = None
        if profile is None:
            block_ids = np.unique(blocks)
            if len(block_ids) <= args.holdout_blocks:
                raise ValueError("Record enough blocks to leave at least one training block.")
            train = blocks < block_ids[-args.holdout_blocks]
            model = fit_decoder(eeg[train], labels[train], data["fs"], components=None,
                                n_classes=len(MOVE_NAMES), baseline_eeg=data["baseline"][train])
        else:
            from .personalization import PlayerDecoder

            if profile.resolve() in {args.session.resolve(), args.out.resolve()}:
                raise ValueError("Use different files for the profile, seed session, and round log.")
            with args.session.open("rb") as archive:
                digest = file_digest(archive, "sha256")
            digest.update(bytes([args.imu]))
            seed_features = decoder_features(eeg, data["fs"])
            decoder = stack.enter_context(
                PlayerDecoder(profile, seed_features, labels, signature=digest.hexdigest())
            )

        print("Circle beats Triangle. Triangle beats Square. Square beats Circle.")
        print("At GO, choose a shape in your mind and picture it without moving your eyes.")
        if decoder is None:
            print("The game scores the decoded move. You'll report your intended move before seeing the result.")
            print("Use the same wearer, electrodes, and board inputs as in the recording.", flush=True)
        else:
            print("Confirm or correct the EEG prediction. Your confirmed move is scored and teaches the decoder.")
            print("Use a separate profile for each wearer. Keep electrode placement and input order "
                  "consistent with the seed recording.", flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        board = stack.enter_context(NeuroPawn(args.port, imu=args.imu, inputs=data["inputs"], gain=data["gain"]))
        if board.fs != data["fs"]:
            raise ValueError("The headset sample rate differs from the recorded session.")
        log = stack.enter_context(args.out.open("x", encoding="utf-8"))
        for number in range(1, args.rounds + 1):
            input(f"\nRound {number}/{args.rounds}. Press Enter and rest until GO. ")
            opponent = randbelow(len(MOVE_NAMES))
            entry = {"round": number, "opponent": MOVE_NAMES[opponent]}
            if decoder is not None:
                entry.update(mode="adaptive", round_id=uuid4().hex)
            try:
                baseline, imagery = capture_move(
                    board, seconds=eeg.shape[-1] / board.fs, skip=data["skip"],
                    baseline_seconds=data["baseline"].shape[-1] / board.fs,
                )
            except ADCRailError:
                clipped += 1
                print("\nEEG clipped. Round skipped. Check electrode contact and the ear clip.")
                entry["status"] = "clipped"
            else:
                if decoder is None:
                    features = decoder_features(imagery[None], board.fs, baseline_eeg=baseline[None])
                    scores = model.predict_proba(features)[0]
                    best = int(np.argmax(scores))
                    decoded = int(model.classes_[best])
                    score = float(scores[best])
                    # Keep the EEG prediction hidden until the blind report.
                    intended = report_move()
                else:
                    features = decoder_features(imagery[None], board.fs)[0]
                    prediction = decoder.predict(features)
                    decoded, score = prediction.move, prediction.confidence
                    entry.update(seed_decoded=MOVE_NAMES[prediction.seed_move],
                                 seed_score=prediction.seed_confidence,
                                 candidate_decoded=MOVE_NAMES[prediction.candidate_move],
                                 candidate_score=prediction.candidate_confidence,
                                 using_player=prediction.using_player,
                                 examples_before=prediction.examples)
                    source = "personalized" if prediction.using_player else "seed"
                    print(f"EEG predicts {MOVE_NAMES[decoded]} "
                          f"({score:.0%} uncalibrated confidence from the {source} model).")
                    intended = _confirm_move(decoded)

                entry.update(decoded=MOVE_NAMES[decoded], score=score)
                if intended is None:
                    unlabeled += 1
                    entry["status"] = "unlabeled"
                    print("Round skipped without scoring or teaching.")
                else:
                    played = decoded
                    if decoder is not None:
                        examples = decoder.teach(entry["round_id"], features, intended, prediction)
                        played = intended
                        seed_match = intended == prediction.seed_move
                        candidate_match = intended == prediction.candidate_move
                        seed_hits += seed_match
                        candidate_hits += candidate_match
                        candidate_shown += prediction.using_player
                        entry.update(played=MOVE_NAMES[played], seed_match=seed_match,
                                     candidate_match=candidate_match, examples_after=examples)
                    result = round_result(played, opponent)
                    match = intended == decoded
                    counts[intended] += 1
                    hits[intended] += match
                    outcomes[result] += 1
                    entry.update(intended=MOVE_NAMES[intended], match=match, result=result)
                    if decoder is None:
                        print(f"EEG chose {MOVE_NAMES[decoded]} ({score:.0%} model score). "
                              f"Computer chose {MOVE_NAMES[opponent]}. {result}.")
                        print(f"You reported {MOVE_NAMES[intended]}. "
                              f"{'Match.' if match else 'Different from the EEG choice.'}")
                    else:
                        print(f"You played {MOVE_NAMES[played]}. Computer chose {MOVE_NAMES[opponent]}. {result}.")
                        print(f"{'Prediction matched.' if match else 'Prediction corrected.'} "
                              f"Profile has {examples} confirmed rounds.")
            print(json.dumps(entry), file=log, flush=True)

    print(f"\n{outcomes['Win']} wins · {outcomes['Loss']} losses · {outcomes['Draw']} draws")
    if profile is None:
        print(f"EEG matched {hits.sum()} of {counts.sum()} reported moves. "
              f"{clipped} of {args.rounds} rounds skipped for clipping.")
    else:
        print(f"Before feedback, shown predictions matched {hits.sum()} of {counts.sum()} confirmed moves. "
              f"The frozen seed matched {seed_hits} and the personalized candidate matched {candidate_hits}.")
        print(f"The personalized candidate was shown in {candidate_shown} labeled rounds.")
        print(f"Clipped rounds {clipped} · Unlabeled rounds {unlabeled}")
    for name, correct, total in zip(MOVE_NAMES, hits, counts):
        print(f"{name} matches {correct}/{total}" if total else f"{name} was never reported.")
    if np.all(counts):
        print(f"Mean recall across the three moves {(hits / counts).mean():.1%} "
              f"(uniform guessing {1 / len(MOVE_NAMES):.1%}).")
    print(f"Round log saved to {args.out}.")
