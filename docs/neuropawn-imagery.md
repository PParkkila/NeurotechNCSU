# NeurotechNCSU

NeuroPawn imagery prototype for the Fall 2026 NeuroTech event at NCSU. Play a three-move game by privately imagining a circle, triangle, or square. The prototype also supports the original two-layout drawing experiment.

## Setup

Use Python 3.11–3.14. The software workflow was checked on Python 3.14. If your default `python3` is newer, choose a supported interpreter explicitly when creating the environment.

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Record a 150-trial dataset in the terminal

Plug in the board, close any apps using its serial port, and run these commands from the project folder. Replace the example port and list only the inputs wired to your posterior electrodes, in the order you want saved. The example uses all eight. Keep the reference connected. Gain defaults to 4. Add `--imu` for the Knight Board IMU.

```sh
python -m imagery record-rps150 --port /dev/cu.usbserial1234 --inputs 1,2,3,4,5,6,7,8
python -m imagery export-rps150
```

Press Enter to start each block of 15. Rest at `+`, read the shape name when it flashes, then picture that shape and hold still at **GO**. The next trial starts automatically. Each block shuffles five circles, five squares, and five triangles. Ten blocks give 50 of each shape and take about 21 minutes of capture time, plus breaks. Each trial saves two seconds of rest before the cue and four seconds of imagery starting one second after GO.

The recorder saves completed blocks to `runs/rps-150.npz`. To continue an interrupted recording, rerun the same command with `--resume` and the original headset setup. `export-rps150` creates `runs/rps-150.json` for the browser. Both files contain labeled EEG and stay in the local `runs/` folder.

1. Open Shape Battle in desktop Chrome. Under **Board settings**, enter the same ordered inputs and gain, then click **Connect NeuroPawn**. Keep the electrode placement and wearer the same.
2. Under **Decoder setup**, expand **Use a saved dataset**, select `runs/rps-150.json`, and click **Train from dataset**.
3. Start a brainwave read and privately imagine any shape at **GO**. Confirm the guess or select the shape you actually imagined before pressing **Battle**. The game uses confirmed rounds as feedback for later reads. Use **Resume previous session** in the same browser to keep that feedback.

To use this workflow locally before deploying the updated page, run `python3 scripts/build_web.py` and then `python3 -m http.server 8000 --bind 127.0.0.1 --directory web-dist` from the project folder. Open `http://127.0.0.1:8000` in desktop Chrome and connect over Web Serial.

## Teach the decoder in Shape Battle

Open the deployed HTTPS page in desktop Chrome on the laptop connected to the NeuroPawn. Close EXG Visualizer or other apps using the board. Click **Connect NeuroPawn**, select its USB serial port, and wait about 30 seconds for the board to configure. The default is gain 4 and inputs 1, 2, and 6. Open **Board settings** before connecting to match your wiring. Keep the game tab visible during each EEG read. [Chrome Web Serial](https://developer.chrome.com/docs/capabilities/serial)

1. Under **Decoder setup**, select **Start training**. The browser runs 150 trials automatically, with 50 prompts for each shape. Rest at `+`, read the shape name, then picture it at **GO**. The full sequence takes about 20 minutes. Use **Pause after this trial** and **Continue training** to take a break. The browser saves each completed trial so you can resume in the same browser after reconnecting.
2. The browser scores the last 30 cued trials using a decoder trained on the first 120, then fits the starting model on all 150 using every selected input. In gameplay, privately choose a shape at **GO**. Review the EEG guess and its three **Shape comparisons**, then confirm or correct it before pressing **Battle**. The computer independently picks each shape with probability 1/3.
3. The dashboard updates at confirmation. It scores the original guess against your confirmed label and shows actual and predicted counts per shape. Macro precision averages shapes the model has predicted. Macro recall averages shapes the player has confirmed. An unavailable value appears as a dash. The dashboard excludes calibration and discarded reads.

The browser decoder trains on the chosen dataset or the current session's calibration. It extracts Fourier band powers with Welch's method and projects them into up to five principal components. In 150-trial sessions, it subtracts resting log power on each channel. That also cancels voltage scaling differences for imported datasets. Three binary models compare circle vs square, triangle vs circle, and square vs triangle. Their log odds are combined into one distribution over shapes. From the first confirmation, simulated annealing searches for a bounded update on prior feedback and the training data. A feedback label has one-quarter the training weight of an initial label. [Simulated annealing](https://doi.org/10.1126/SCIENCE.220.4598.671)

The confidence estimate uses the two held-out blocks for 150-trial sessions or leave-one-out predictions for older browser sessions, alongside prior gameplay predictions made before the confirmed label was known. Temperature scaling and shrinkage toward equal probabilities limit confidence when evidence is sparse. [Temperature scaling](https://proceedings.mlr.press/v70/guo17a.html)

The Vercel page records two seconds of rest, displays each training cue for one second, then records five seconds after GO. The decoder uses seconds 1–5 after GO. The browser saves one IndexedDB record per session, with board settings, raw EEG, training labels, predictions, corrections, and computer moves. Use **Download session data** to save the labeled recording and feedback as JSON. Reload the same site in the same browser, reconnect with the same board settings, and select **Resume previous session**. Select **Start training** for a new wearer. [Knight Board data format](https://docs.neuropawn.tech/knight-board/data-format/) · [Knight Board command set](https://docs.neuropawn.tech/knight-board/command-set/)

If one input clips during rest or imagery, the browser records which input clipped and leaves it out of the decoder for that whole trial. A second clipped input in the same trial triggers a retry. A single-input setup also requires a retry if that input clips.

### Local Python bridge

For the local bridge, open **Use the local Python bridge** on the hosted page and follow its download instructions, or run the command below from the project folder on the laptop connected to NeuroPawn.

```sh
python -m imagery web --port /dev/cu.usbserial1234
```

Open the local address printed by the bridge. Pass `--inputs` and `--gain` to match the board wiring. The Python bridge uses its original 20-read decoder and saves a local `.npz` per session under `runs/web-sessions/` by default. Use `--sessions` to select a different directory.

```python
import json
import numpy as np

with np.load("runs/web-sessions/SESSION_ID.npz", allow_pickle=False) as session:
    metadata = json.loads(session["metadata"].item())
    rest = session["baseline"]
    imagery = session["eeg"]
```

For Vercel Drop, run `python3 scripts/build_web.py` from the project folder and upload `web-dist/`.

## Play and teach a personal decoder

**Circle beats triangle. Triangle beats square. Square beats circle.** Matching shapes draw.

```sh
python -m imagery play --session runs/rps.npz \
  --port /dev/cu.usbserial1234 --profile runs/player.sqlite3
```

Replace the example port with your board's port. Use a labeled three-shape recording as the seed. The recording command below creates `runs/rps.npz`. Keep the electrode placement and input order consistent for each player profile. Use a separate `.sqlite3` file for each player or headset setup.

At GO, privately choose and imagine one of the three shapes. The game shows its prediction and a percentage model score. Press Enter to confirm it, type your actual shape to correct it, or type `skip` to omit that round. The confirmed shape determines the round against the computer and adds a training example. Reuse the same profile path to continue learning across sessions. Each launch writes a separate round log in `runs/` by default.

The initial decoder is a regularized classifier trained with Fourier features and five SVD components on the supplied seed trials. During teaching, the scaler and SVD projection stay fixed. The classifier refits on the seed and confirmed player rounds. Before class reweighting, the seed carries 18 equivalent examples and each confirmed round adds one-quarter of an example. The refit keeps the seed's class weights and gradually blends the learned scores into the seed scores. The player database stores features, labels, and pre-feedback predictions locally.

The program compares the learning model with a frozen seed model on each round before the player reports the move. It displays the seed prediction until at least 12 moves have been confirmed under the current learning policy, with at least three of each shape, and the blended prediction has gained at least two correct predictions and 10 percentage points of mean recall across the three shapes. The personalized share changes by at most five percentage points per confirmed round. It gradually returns to the seed if the advantage disappears. Existing labeled examples still train the model. The serving decision uses fresh comparisons under the slower policy.

The log records the displayed prediction and both models' pre-feedback scores. Scores are uncalibrated classifier probabilities. Available validation is inconclusive for privately chosen shapes. To assess personalization, compare both predictions on fresh rounds and separately test a frozen model with the prediction hidden until after the report.

Supervised feedback and a seed decoder follow the coadaptive design tested by Vidaurre and colleagues in sensorimotor imagery. EEG transfer studies also report substantial wearer differences. The weighting above is a prototype choice for this recording. [Coadaptive calibration](https://pubmed.ncbi.nlm.nih.gov/21162666/) · [EEG transfer learning](https://pubmed.ncbi.nlm.nih.gov/31034407/)

## Record and evaluate imagined shapes

**Circle beats triangle. Triangle beats square. Square beats circle.** Two of the same shape draw.

Look at the fixation cross during rest. At **GO**, choose a shape and hold its image in your mind while keeping your gaze steady. The screen shows the same prompts for every move. After recording ends, type the shape you imagined. During play, the decoder has already picked your move by that point. The game uses the decoded move against a random computer move and separately checks whether the decoder matched what you reported.

The default montage uses gain 4 and board inputs **1, 2, 3, 6** at **O1, Oz, O2, POz**, respectively. Keep the reference connected and use the same placement, gain, and wearer for both sessions.

```sh
python -m imagery record --task rps --port /dev/cu.usbserial1234 --out runs/rps.npz
python -m imagery evaluate --session runs/rps.npz --out runs/rps-report.json --svg runs/rps-held-out.svg
python -m imagery play --session runs/rps.npz --port /dev/cu.usbserial1234 --out runs/rps-game.jsonl
```

Calibration defaults to five blocks of 12 clean trials. Within each block, use each shape four times in mixed order. Each trial saves two seconds of rest and four seconds of imagery after the first second at GO. Completed blocks are saved as you go. For a software-only check, use `simulate --task rps --out runs/rps-synthetic.npz` and evaluate that file.

`evaluate` compares raw Fourier features, Fourier plus SVD, and Fourier features with each trial's resting log-power subtracted. All use the same last two complete blocks as the holdout. The report also includes a pre-GO rest-only control to check whether the choices were predictable before imagery began. Balanced three-way guessing is 33⅓%. `play` uses the rest-subtracted decoder fitted to the earlier blocks and logs fresh rounds. The software has been tested with simulated inputs. Decoding accuracy on real, privately chosen shapes is still unknown.

### Try it without hardware

```sh
python -m imagery simulate
python -m imagery evaluate --session runs/synthetic.npz \
  --out runs/synthetic-report.json --svg runs/synthetic-layout.svg --open
```

The synthetic recording has a deliberately injected signal. It tests the software path and is labeled as synthetic in the JSON and SVG.

### Record, compare, and draw with one headset

1. Fit the reference and electrodes according to the [NeuroPawn setup guide](https://docs.neuropawn.tech/getting-started/). Wire physical inputs **1–6** to **O1, Oz, O2, PO3, PO4, POz**, in that order. Wire **7–8** to **Fp1, Fp2** for a frontal comparison. Use `record --channels 6` if you only fit the posterior electrodes.
2. Connect the board over USB. Identify the new `/dev/cu.*` port on macOS or the `COM` port on Windows. Check signal quality in EXG Visualizer, then close it to free the serial port. Use the default 125 Hz board configuration. Add `--imu` to both hardware commands for the Knight Board IMU.
3. Run these commands from the repo. Replace the example port with your board's actual port.

```sh
python -m imagery record --port /dev/cu.usbmodem1234
python -m imagery evaluate --open
python -m imagery live --port /dev/cu.usbmodem1234 --open
```

`record` defaults to five balanced blocks of 20 trials. **A** means circle above square. **B** means square above circle. Look at the center cross after the letter disappears. At **GO**, imagine the layout and keep your gaze and posture steady. Take breaks at the block prompts. Completed blocks are saved to `runs/session.npz` as you go. Use `--out` with a fresh filename for another session, and pass the same path as `--session` when evaluating or drawing.

For a simpler task, pass `record --task shape --out runs/shapes.npz` along with your port. The cues are **CIRCLE** and **SQUARE**. Imagine the single shape in the center at **GO**. Pass `--session runs/shapes.npz` to `evaluate` or `live` to draw that task's predictions. Keep the two tasks in separate sessions.

For a recent-image recall test, use `record --task shape-preview --out runs/shape-preview.npz`. A shape appears for one second, then disappears. Picture it again at **GO**. This session saves the preview task separately from the word-cue task, and `evaluate` draws the predicted shape.

If a short contact check shows clipping, check the electrodes and shared reference first. To test a lower gain on a new recording, use `record --gain 4` with your usual arguments. The recorder stores the gain and `live` reuses it. For an older recording without gain metadata, pass its actual gain explicitly to `live` if it differs from 12. [NeuroPawn clipping troubleshooting](https://docs.neuropawn.tech/troubleshooting/#trace-is-clipped-flat-at-a-limit-or-unusually-large)

`evaluate` saves `runs/report.json` with every held-out prediction and `runs/held-out.svg` with the **first** held-out SVD prediction and its target. `live` refits on the earlier blocks, uses the same feature pipeline for a fresh recording, and saves `runs/live.svg`. Use the same wearer and channel placement for both recordings. In live mode, choose A or B privately, press Enter, then start imagining at **GO**. The displayed model score is an uncalibrated classifier probability.

## What the comparison tests

- Each trial contributes one feature row. From about 0.75–3.75 seconds after GO, Welch estimates log power in four bands across the six posterior channels. The frontal channels, when present, get a separate score.
- Both models use regularized logistic regression. One uses all 24 standardized features. The other uses five PCA components from a full SVD. Scaling, SVD, and model fitting use only the earlier blocks. The last two whole blocks are scored once by default.
- The report includes paired correct counts and the SVD accuracy difference in percentage points. Within-training-block label shuffles provide a chance comparison for each model. Those shuffle scores do not test whether SVD's improvement itself is statistically significant.
- Fix `--components` and `--holdout-blocks` before evaluating your final session. A positive result is a two-layout classification result for that recording protocol. The frontal comparison can flag a confound, and a fresh session with altered cues gives a stronger follow-up test.

`runs/`, player profiles, and other `.npz` recordings are ignored by Git.

## EEG-guided drawing with visible targets

Use this separate mode to validate an EEG input and deliberately select a shape. Connect **O1, Oz, and O2** to board inputs **1, 2, and 3**. Keep the ear reference connected. Turn off unused inputs using the board switches. The software also removes unused inputs from the RLD network and powers their channels off.

```sh
python -m imagery select --port /dev/cu.usbserial1234 --open
```

This command supports the standard Knight Board and defaults to gain 4. It opens a local browser page. The flicker starts when you acknowledge the notice and click **Start trial**. Use a steady 60 Hz or 120 Hz display, keep the browser focused, and use Esc to stop the flicker. The browser checks its frame timing in software.

During nine calibration trials, look at the indicated target or the center cross. The app then locks its selection gate and runs 12 shape trials plus four idle trials. It counts abstentions as misses and unlocks drawing after at least 10 of 12 correct and zero idle selections in that pilot. The report saves scores and outcomes locally, without raw EEG. This is an SSVEP selection task. The shape-recall experiment above has its own session and decoder. [EEG-guided drawing paper](https://www.nature.com/articles/s44385-026-00086-6) · [NeuroPawn SSVEP guide](https://docs.neuropawn.tech/tutorials/ssvep/)
