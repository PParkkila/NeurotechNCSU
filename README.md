# NeuroPawn Shape Battle

Shape Battle is a three-choice EEG game built for the Fall 2026 NeuroTech Hackathon at NCSU. A player imagines a **circle**, **triangle**, or **square**. The browser turns the NeuroPawn Knight Board signal into frequency features, predicts the shape, and asks the player to confirm or correct the guess. The confirmed shape plays against a uniformly random computer move and becomes a labeled example for later rounds.

In the browser, **square beats triangle**, **triangle beats circle**, and **circle beats square**. Matching shapes draw. [Browser rules](config.js)

## Run the website

1. Plug in the NeuroPawn over USB and close any other app using its serial port. Open the hosted page in desktop Chrome. Under **Board settings**, enter the board inputs in acquisition order and set the gain to match the headset. The browser defaults to inputs 1, 2, and 6 at gain 4. Click **Connect NeuroPawn** and select the board's serial port. [^webserial] [^setup] [^commands]
2. Select **Start training**. Rest at **+**, read the shape cue, then hold the shape in your mind at **GO**. The 150 trials advance automatically and take about 20 minutes plus breaks. Use **Pause after this trial** and **Continue training** to take a break. The browser saves each completed trial.
3. After training, start a brainwave read and privately choose a shape at **GO**. Confirm or correct the prediction before pressing **Battle**. Use **Download session data** to export the EEG, labels, and feedback as JSON. To continue in the same browser, reconnect with the same settings and select **Resume previous session**.

Build and serve the same static site locally with Python, then open [the local website](http://127.0.0.1:8000) in desktop Chrome.

```sh
python3 scripts/build_web.py
python3 -m http.server 8000 --bind 127.0.0.1 --directory web-dist
```

For hosting, connect the repository to Vercel using the checked-in build configuration, or upload the generated `web-dist/` folder with Vercel Drop. The local Python bridge, terminal recording, and saved-dataset import are also available. [Vercel configuration](vercel.json) · [Local bridge](docs/local-bridge.md) · [Recording and import instructions](docs/neuropawn-imagery.md#record-a-150-trial-dataset-in-the-terminal) [^vercel] [^drop]

## How the browser decoder works

### 1. Collect a balanced dataset

The Knight Board streams EEG at $f_s=125$ Hz. Each prompted trial records two seconds of rest, displays a one-second shape cue, then records five seconds of imagery after **GO**. The decoder skips the first second after GO, leaving 250 rest samples and 500 analyzed imagery samples per input. Ten shuffled blocks of 15 trials yield **50 examples per shape**. Each block contains five of each shape. [Acquisition code](browser-serial.js) · [Trial schedule and capture](browser-game.js) [^frames]

If one input clips during rest or imagery, the browser records its input number and masks that channel for the entire trial. It requests a retry if a second input clips or if the only active input clips. The browser stores the session in IndexedDB, including the board settings, raw EEG, prompt labels, and gameplay feedback. [Clipping detection](browser-serial.js) · [Session storage](browser-store.js) [^clipping]

### 2. Convert EEG to rest-normalized Fourier features

For each channel $c$, `logPowers()` divides the signal into one-second windows of $N=125$ samples with a 63-sample hop. For segment $j$, it subtracts the segment mean $\bar x_{c,j}$ and multiplies by the periodic Hann window $w[n]=\tfrac12(1-\cos(2\pi n/N))$. The discrete Fourier coefficient at bin $k$ is

$$
X_{c,j}(k)=\sum_{n=0}^{N-1} w[n]\bigl(x_{c,j}[n]-\bar x_{c,j}\bigr)e^{-2\pi i kn/N}.
$$

Welch's method averages the windowed periodograms, which are proportional to $|X_{c,j}(k)|^2$, to estimate the power spectral density $S_c(k)$. At 1 Hz bin spacing, the four bands sum bins **4–7**, **8–12**, **13–19**, and **20–30 Hz**. For band $B$, define $P_{c,B}=\sum_{k\in B}S_c(k)$. The feature is the log-power difference $f_{c,B}=\log\max(P^{\mathrm{go}}_{c,B},10^{-12})-\log\max(P^{\mathrm{rest}}_{c,B},10^{-12})$. This measures the change from that trial's own rest period and gives four features per selected input. [`logPowers()` and `features()`](browser-decoder.js) [^welch]

### 3. Standardize and project with PCA

For each feature column $d$, `projector()` learns a mean $\mu_d$ and scale $\sigma_d$ from the training rows, then standardizes each feature as $z_d=(f_d-\mu_d)/\sigma_d$. A clipped channel's missing features contribute zero in standardized space, equivalent to filling them with the training mean. Features with no observed training values contribute zero even if that input returns during play.

The code forms the covariance $C=Z^\top Z/n$ from the $n$ standardized training rows and uses Jacobi rotations to find its eigenvectors. It retains up to five directions of largest variance. The projected scores are $u=V_K^\top z$. Their norm is capped at 3, and a constant 1 is appended for the bias term, giving the classifier input $h$. [`projector()`](browser-decoder.js) [^pca]

### 4. Train three pairwise logistic classifiers

The three comparisons are **circle vs. square**, **triangle vs. circle**, and **square vs. triangle**. Each sees only the trials labeled with one of its two shapes. For pair $(a,b)$, the model computes a margin $m_{ab}(h)=\theta_{ab}^\top h$ and a pairwise probability $P(a\mid a,b,h)=\sigma(m_{ab}(h))$, where $\sigma(t)=1/(1+e^{-t})$.

`initialWeights()` fits each classifier by momentum gradient descent on weighted logistic loss with an $L_2$ penalty. With example label $y_i=1$ for the first shape in the pair, margin $m_i=\theta^\top h_i$, and example weight $\omega_i$, the objective is

$$
L(\theta)=\sum_i \omega_i\bigl[\log(1+e^{m_i})-y_i m_i\bigr]+\tfrac12\sum_d\lambda_d\theta_d^2.
$$

The browser uses $\lambda_d=1$ for feature coefficients and $\lambda_d=0.25$ for the bias. To combine the comparisons, `couple()` treats the margins as signed log-odds with $m_{ba}=-m_{ab}$. The centered class scores $s_a=\tfrac13\sum_{b\ne a}m_{ab}$ are the least-squares fit to the three pairwise score differences. Softmax converts them to one distribution, $p_a=e^{s_a}/\sum_b e^{s_b}$. The largest $p_a$ determines the guess. [`loss()`, `initialWeights()`, and `couple()`](browser-decoder.js) [^logistic]

### 5. Evaluate, refit, and learn from feedback

- **Hold out the final blocks.** `validateHoldout()` fits the feature transform and classifiers on blocks 1–8, then scores blocks 9–10. This gives a 120-trial training split and a 30-trial cued holdout. The game fits its starting model on all 150 trials after scoring the holdout. [Training and holdout](browser-game.js) · [Decoder](browser-decoder.js)
- **Update after confirmation.** Each prompt has training weight $1$, and each confirmed gameplay example has weight $0.25$. The PCA projection stays anchored to the initial dataset. `anneal()` searches nearby classifier weights using the original trials and confirmed rounds. For a proposed loss increase $\Delta L$, the acceptance probability is $e^{-\Delta L/T}$ at temperature $T$. Proposals stay within distance $0.12/\sqrt{10}$ of the previous coefficients, and the best candidate found is used for the next read. [`anneal()` and `boundedProposal()`](browser-decoder.js) [^annealing]
- **Calibrate confidence separately.** The browser fits inverse temperature $\alpha=1/T$ on held-out prompted predictions and earlier gameplay predictions saved before their labels were known. The scaling function is $q_a(\alpha)=p_a^\alpha/\sum_b p_b^\alpha$, with a small numerical floor on $p_a$. Prompted reads have confidence-fit weight $0.5$, and confirmed gameplay reads have weight $1$. The displayed distribution mixes $q_a$ with $1/3$ in proportion $\rho=2/(3+0.5n_{\mathrm{hold}}+n_{\mathrm{play}})$ toward uniform confidence. The highest-scoring shape stays the same. [Confidence implementation](browser-confidence.js) [^temperature]
- **Score the original prediction.** The dashboard compares each pre-feedback guess with the player's confirmed label. For shape $a$, precision is $\mathrm{TP}_a/(\mathrm{TP}_a+\mathrm{FP}_a)$ and recall is $\mathrm{TP}_a/(\mathrm{TP}_a+\mathrm{FN}_a)$. Macro averages use only shapes with a defined denominator. Training trials and discarded reads are excluded. [Gameplay metrics](browser-game.js)

## Evaluation scope

The repository also includes a Python teaching prototype with a different adaptation policy and a separate flicker-based SSVEP drawing experiment. Coadaptive BCI and EEG transfer studies inform the context for wearer-specific training. The SSVEP experiment uses visual targets and a separate decoder. [Python teaching and SSVEP protocols](docs/neuropawn-imagery.md) [^coadaptive] [^transfer] [^drawing] [^ssvep]

## Limitations and potential improvements

| Limitation | What to try next |
| --- | --- |
| The browser holdout uses prompts from the same recording. Shape cues, eye movements, and trial order can all correlate with the labels. | Run a separate session with neutral cues and privately chosen shapes. Record the intended shape before revealing the guess. Keep a rest-only control and report its score alongside imagery accuracy. |
| Early validation covered one wearer and one recording session. | Record multiple wearers across different days. Hold out entire sessions to test repeatability, and entire wearers to test whether training transfers between people. Report per-wearer results and uncertainty alongside the aggregate. |
| The model uses four band powers per channel and ignores one clipped input per trial. Movement, poor contact, or interference can affect samples even when they stay within the board's range. | Track contact quality and rejection rates per input. Compare electrode subsets and artifact screening on the same held-out trials before changing the decoder. |
| The output is restricted to the three trained shapes. Band powers summarize the analyzed window and discard timing and phase. | Compare the current features against time-resolved band powers or features that use relationships between electrodes. Keep the same holdout and compare against the simpler model before expanding the shape set. |
| The adaptive model uses labels reported after the player sees the guess, and confidence estimates start with relatively little evidence. | Score a frozen model and the adaptive model side by side on the same fresh rounds. Report per-shape recall and compare confidence estimates with observed accuracy as more blind trials accumulate. |

## Repository layout

| Path | Contents |
| --- | --- |
| [`index.html`](index.html), [`session-game.js`](session-game.js) | Training screens, gameplay, and prediction dashboard |
| [`browser-serial.js`](browser-serial.js), [`browser-store.js`](browser-store.js) | USB EEG acquisition and browser-local session storage |
| [`browser-game.js`](browser-game.js), [`browser-decoder.js`](browser-decoder.js), [`browser-confidence.js`](browser-confidence.js) | Trial scheduling, feature extraction, model fitting, feedback learning, and confidence |
| [`imagery/`](imagery/) | Python recording, offline evaluation, terminal play, local bridge, and drawing experiments |
| [`docs/`](docs/), [`scripts/build_web.py`](scripts/build_web.py) | Setup instructions and static site packaging |

## Credits

**Team** — Ishan Hitesh Mistry, Khrystyna Lukecha, Patrik Parkkila, Natalia Bernardo. The team defined the project goals, directed the design, built and wired the hardware, tested the system, and integrated and debugged the software.

**AI assistance** — Claude (Anthropic) assisted with the game dashboard. Codex (OpenAI) assisted with game development, prototyping, data collection, and calibration.

**Tools** — NeuroPawn Knight Board, BrainFlow, Python, JavaScript, and Vercel. [License](LICENSE)

[^webserial]: Chrome for Developers. [Read from and write to a serial port](https://developer.chrome.com/docs/capabilities/serial).
[^setup]: NeuroPawn. [Getting Started with the NeuroPawn Kit](https://docs.neuropawn.tech/getting-started/).
[^commands]: NeuroPawn. [Knight Board Command Set](https://docs.neuropawn.tech/knight-board/command-set/).
[^vercel]: Vercel. [Static Configuration with vercel.json](https://vercel.com/docs/project-configuration/vercel-json).
[^drop]: Vercel. [Vercel Drop](https://vercel.com/kb/vercel-drop).
[^frames]: NeuroPawn. [Knight Board Data Format](https://docs.neuropawn.tech/knight-board/data-format/).
[^clipping]: NeuroPawn. [Troubleshooting clipped or flat traces](https://docs.neuropawn.tech/troubleshooting/#trace-is-clipped-flat-at-a-limit-or-unusually-large).
[^welch]: P. D. Welch (1967). [The use of fast Fourier transform for the estimation of power spectra](https://doi.org/10.1109/TAU.1967.1161901). *IEEE Transactions on Audio and Electroacoustics*, 15(2), 70–73.
[^pca]: I. T. Jolliffe and J. Cadima (2016). [Principal component analysis: a review and recent developments](https://doi.org/10.1098/rsta.2015.0202). *Philosophical Transactions of the Royal Society A*, 374, 20150202.
[^logistic]: scikit-learn. [Logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression).
[^annealing]: S. Kirkpatrick, C. D. Gelatt Jr., and M. P. Vecchi (1983). [Optimization by Simulated Annealing](https://doi.org/10.1126/science.220.4598.671). *Science*, 220(4598), 671–680.
[^temperature]: C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger (2017). [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html). *Proceedings of ICML*, 1321–1330.
[^coadaptive]: C. Vidaurre, C. Sannelli, K.-R. Müller, and B. Blankertz (2011). [Machine-learning-based coadaptive calibration for brain-computer interfaces](https://pubmed.ncbi.nlm.nih.gov/21162666/). *Neural Computation*, 23(3), 791–816.
[^transfer]: H. He and D. Wu (2020). [Transfer Learning for Brain–Computer Interfaces: A Euclidean Space Data Alignment Approach](https://pubmed.ncbi.nlm.nih.gov/31034407/). *IEEE Transactions on Biomedical Engineering*, 67(2), 399–410.
[^drawing]: G. Wang, Y. Huang, L. Muckli, and D. Faccio (2026). [Symbiotic brain-machine drawing via visual brain-computer interfaces](https://www.nature.com/articles/s44385-026-00086-6). *npj Biomedical Innovations*, 3, 31.
[^ssvep]: NeuroPawn. [SSVEP tutorial](https://docs.neuropawn.tech/tutorials/ssvep/).
