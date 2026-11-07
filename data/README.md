# Three-shape seed recording

`neuropawn-rps-seed.npz` contains 60 labeled trials from one wearer and one NeuroPawn Knight session. Each of five chronological blocks contains four trials per shape. The wearer chose privately at GO and reported the intended shape after EEG capture. Block 4 grouped trials by shape, and block 5 repeated a three-shape sequence.

| Field | Contents |
| --- | --- |
| `eeg` | 60 trials × 3 inputs × 500 saved imagery samples |
| `baseline` | 60 trials × 3 inputs × 250 pre-GO rest samples |
| `labels` | 0 circle, 1 triangle, 2 square |
| `blocks` | Chronological block IDs 1–5 |
| `inputs` | Board inputs 1, 2, and 6, in array order. Scalp labels after lead swaps were unverified. |
| `fs`, `gain`, `skip` | 125 Hz, gain 4, first 1 second after GO omitted from saved imagery |
| `source`, `task` | `neuropawn`, `rps` |

Load the archive with `numpy.load(path, allow_pickle=False)`. The file contains raw EEG and self-reported shape labels. It is the single recording included as seed data. Player feedback profiles and subsequent recordings stay local.

The original held-out split trained on blocks 1–3 and scored blocks 4–5. Fourier + SVD scored 12/24 with shuffled-label p = 0.076. The teaching mode uses all 60 trials as a starting point. Evaluate its performance on new rounds, including a separate blind session for an accuracy claim.
