// Local, calibrated play against the computer.
(function () {
  const $ = (id) => document.getElementById(id);
  const escape = (value) => String(value ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const cap = (shape) => shape ? shape[0].toUpperCase() + shape.slice(1) : "";
  const percent = (value, digits = 0) => Number.isFinite(value) ? (100 * value).toFixed(digits) + "%" : "—";
  const progress = (seconds) => `<div class="progress"><i style="animation-duration:${seconds}s"></i></div>`;
  const back = (icon, text, locked = false) =>
    `<div class="back ${locked ? "locked" : ""}"><div>${icon}</div><small>${text}</small></div>`;
  const metrics = [
    ["accuracy", "Accuracy", "var(--good)", ""],
    ["macro_recall", "Macro recall", "var(--warn)", "5 3"],
    ["macro_precision", "Macro precision", "var(--text)", "2 3"]
  ];

  function trend(history) {
    if (!history.length) return "";
    const rounds = history.map((point) => point.round);
    const last = rounds.at(-1);
    const x = (round) => last === 1 ? 186 : 44 + (round - 1) / (last - 1) * 284;
    const y = (value) => 88 - value * 70;
    const grid = [0, 0.5, 1].map((value) =>
      `<line x1="44" x2="328" y1="${y(value)}" y2="${y(value)}" stroke="var(--line)"/>` +
      `<text x="39" y="${y(value) + 3}" text-anchor="end">${value * 100}%</text>`
    ).join("");
    const series = metrics.map(([field, , color, dash]) => {
      const points = history.filter((point) => Number.isFinite(point[field]));
      if (!points.length) return "";
      const path = points.map((point) => `${x(point.round)},${y(point[field])}`).join(" ");
      const end = points.at(-1);
      return `<polyline points="${path}" fill="none" stroke="${color}" stroke-width="2" stroke-dasharray="${dash}"/>` +
        `<circle cx="${x(end.round)}" cy="${y(end[field])}" r="2.5" fill="${color}"/>`;
    }).join("");
    const ticks = last === 1 ? '<text x="186" y="108" text-anchor="middle">Round 1</text>'
      : `<text x="44" y="108">Round 1</text><text x="328" y="108" text-anchor="end">Round ${last}</text>`;
    return `<svg width="100%" height="160" viewBox="0 0 350 118" role="img"` +
      ` aria-label="Cumulative accuracy, macro recall, and macro precision by confirmed round"` +
      ` fill="var(--muted)" font-size="10">${grid}${series}${ticks}</svg>`;
  }

  function start(bridge, EEG) {
    const S = window.NPShapes, B = window.NPBattle;
    const browser = typeof EEG.downloadSession === "function";
    const storageKey = browser ? "np_webserial_calibration_session" : "np_eeg_calibration_session";
    const setup = $("setup"), newPlayer = $("newP1"), newGame = $("btnNew"), resume = $("btnResume");
    let state = null, pick = null, busy = null, capturePhase = "starting", error = "", setupError = "";
    const download = $("btnDownloadSession");
    let savedSession, exporting = false;
    try { savedSession = localStorage.getItem(storageKey); } catch { /* storage unavailable */ }

    $("modePill").textContent = browser ? "● NeuroPawn connected over USB" : "● EEG bridge connected (local game)";
    $("modePill").className = "pill live";
    $("btnRematch").textContent = "Next round";
    $("calibrationIntro").textContent = `Complete ${bridge.calibration_count ?? 20} EEG reads before your first round against the computer.`;

    function accept(next) {
      if (!next || typeof next.id !== "string" || !next.calibration || !next.metrics ||
          !["calibration", "ready", "review", "confirmed", "result"].includes(next.phase) ||
          (["review", "confirmed", "result"].includes(next.phase) && !next.round?.id)) {
        throw new Error("The game returned an invalid session state.");
      }
      if (next.phase !== "review") pick = null;
      else if (state?.round?.id !== next.round.id || state?.phase !== "review") pick = next.round.predicted;
      state = next;
    }

    function slot(id, name, tag, body, active = false, winner = false) {
      const element = $(id);
      element.classList.toggle("active", active);
      element.classList.toggle("winner", winner);
      element.innerHTML = `<div class="slotHead"><h3>${escape(name)}</h3>` +
        `${tag ? `<span class="tag">${tag}</span>` : ""}</div><div class="slotBody">${body}</div>`;
    }

    function captureHTML() {
      if (capturePhase === "rest") return `<div class="big">+</div><p>Rest. Keep your eyes on the cross.</p>${progress(bridge.rest_seconds)}`;
      if (capturePhase === "go") return `<div class="big go">GO</div><p>Imagine your shape. Hold still.</p>${progress(bridge.imagery_seconds)}`;
      if (capturePhase === "decoding") return '<div class="big">…</div><p>Decoding…</p>';
      return '<div class="big">+</div><p>Get ready. Keep your eyes on the cross.</p>';
    }

    function guess(round) {
      const estimate = round.confidenceReads
        ? `Estimated confidence · ${round.confidenceReads} held-out calibration reads · ${round.confidenceFeedback} confirmed rounds`
        : "Uncalibrated model confidence";
      const matches = round.pairwise?.map(({ shapes, probability }) => {
        const [first, second] = shapes.map((index) => cap(S.LIST[index]));
        return `<li>${first} vs ${second} · ${first} ${percent(probability)} · ${second} ${percent(1 - probability)}</li>`;
      }).join("");
      return `<p>EEG predicts <b>${cap(round.predicted)}</b> (${percent(round.score)}).</p>` +
        `<p class="muted tiny">${estimate}</p>` +
        (matches ? `<details class="tiny"><summary>Shape comparisons</summary><ul class="list">${matches}</ul></details>` : "");
    }

    function showShape(shape, size, pop = false) {
      return `<div class="face ${pop ? "pop" : ""}">${S.svg(shape, size)}</div>`;
    }

    function playerHTML() {
      if (busy === "capture") return captureHTML();
      if (busy === "confirm") return '<div class="big">…</div><p>Saving feedback…</p>';
      const { phase, calibration, round } = state;
      if (phase === "calibration") {
        return `<p class="muted">Remember this shape</p><div class="shapeLabel">${cap(calibration.prompt)}</div>` +
          '<button class="primary" data-act="capture">Start calibration read</button>';
      }
      if (phase === "ready") {
        const completed = state.metrics.total === 0
          ? `<p class="muted tiny">${calibration.completed} of ${calibration.total} calibration reads saved</p>` : "";
        return '<p>Imagine any shape when GO appears.</p><button class="primary" data-act="capture">Start brainwave read</button>' + completed;
      }
      if (phase === "review") {
        const buttons = S.LIST.map((shape) =>
          `<button class="shapeBtn ${shape === pick ? "sel" : ""}" data-pick="${shape}" aria-label="${shape}"` +
          ` aria-pressed="${shape === pick}" ${busy ? "disabled" : ""}>${S.svg(shape, 52)}<span>${cap(shape)}</span></button>`
        ).join("");
        return showShape(pick, 100) + guess(round) + '<p>Confirm or correct the shape you imagined.</p>' +
          `<div class="picker">${buttons}</div><div class="row c">` +
          `<button class="primary" data-act="confirm" ${busy ? "disabled" : ""}>Confirm shape</button>` +
          `<button data-act="discard" ${busy ? "disabled" : ""}>Redo</button></div>`;
      }
      const revealed = phase === "result";
      return showShape(round.confirmed, 150, revealed) +
        `<div class="shapeLabel">${cap(round.confirmed)}</div>` + guess(round) +
        (browser ? '<p class="muted tiny">Feedback saved. The decoder will use it on your next read.</p>' : "");
    }

    function renderMetrics() {
      $("sessionMetrics").classList.toggle("hidden", !state || state.phase === "calibration");
      if (!state || state.phase === "calibration") return;
      const m = state.metrics;
      const numbers = metrics.map(([field, label, color]) =>
        `<span><b style="color:${color}">${label}</b> ${percent(m[field], 1)}</span>`).join("");
      const last = m.last_confirmed;
      const latest = last ? `<p>Last confirmed round · EEG guessed <b>${cap(last.predicted)}</b> · you chose <b>${cap(last.actual)}</b></p>` : "";
      const rows = m.per_shape?.map(({ shape, hits, actual, predicted }) =>
        `<tr><th scope="row">${cap(shape)}</th><td>${actual}</td><td>${predicted}</td>` +
        `<td>${percent(predicted ? hits / predicted : null, 1)}</td>` +
        `<td>${percent(actual ? hits / actual : null, 1)}</td></tr>`).join("");
      const detail = rows
        ? `<table class="metricTable tiny"><thead><tr><th scope="col">Shape</th><th scope="col">Actual</th>` +
          `<th scope="col">Predicted</th><th scope="col">Precision</th><th scope="col">Recall</th></tr></thead>` +
          `<tbody>${rows}</tbody></table><p class="muted tiny">Macro averages include shapes with a measured denominator.</p>`
        : "";
      $("sessionMetricsBody").innerHTML = `<div class="row">${numbers}</div>` + trend(m.history ?? []) +
        `<p class="muted tiny">${m.correct} of ${m.total} pre-feedback guesses matched your confirmed shape.</p>` +
        latest + detail +
        (state.file ? `<p class="muted tiny">Local data file <code>${escape(state.file)}</code></p>` : "");
    }

    function render() {
      newGame.disabled = !!busy;
      newPlayer.disabled = !!busy;
      newGame.textContent = state ? "Start new calibration" : "Start calibration";
      resume.classList.toggle("hidden", !!state || !savedSession);
      resume.disabled = !!busy;
      $("setupError").textContent = setupError;
      $("arena").classList.toggle("hidden", !state);
      $("sessionFile").classList.toggle("hidden", !browser || !state);
      download.disabled = !!busy || exporting;
      renderMetrics();
      if (!state) return;

      const { player, phase, calibration, round } = state;
      const capturing = busy === "capture";
      const calibrating = phase === "calibration";
      $("stage").classList.toggle("calibrating", calibrating);
      const revealed = phase === "result" && !capturing;
      const confirmed = phase === "confirmed";
      const winner = revealed ? B.decide(round.confirmed, round.computer) : null;
      const tally = { p1: 0, p2: 0, tie: 0 };
      for (const item of state.results ?? []) tally[B.decide(item.player, item.computer)]++;
      $("codeShow").textContent = calibrating ? "Calibration" : "EEG game";
      $("scoreboard").innerHTML = calibrating ? `${calibration.completed} of ${calibration.total} reads saved`
        : `<b>${escape(player)}</b> ${tally.p1} · ${tally.tie} draw${tally.tie === 1 ? "" : "s"} · ${tally.p2} <b>Computer</b>`;

      const tag = winner === "tie" ? "Draw" : winner === "p1" ? "Winner" : confirmed ? "Locked"
        : calibrating ? "Calibration" : revealed ? "" : "Your turn";
      const feedback = error ? `<p class="err" role="alert">${escape(error)}</p>` : "";
      slot("slot1", player, tag, playerHTML() + feedback, !revealed && !confirmed, winner === "p1");
      const computerLocked = confirmed || phase === "review";
      const computer = revealed
        ? showShape(round.computer, 150, true) + `<div class="shapeLabel">${cap(round.computer)}</div>`
        : back(computerLocked ? "🔒" : "?", calibrating ? "Available after calibration" : "Hidden until Battle", computerLocked);
      slot("slot2", "Computer", winner === "tie" ? "Draw" : winner === "p2" ? "Winner" : computerLocked ? "Locked" : "", computer, false, winner === "p2");

      const banner = $("turnBanner");
      banner.classList.toggle("result", revealed);
      if (revealed) {
        const [win, lose] = winner === "p1" ? [round.confirmed, round.computer] : [round.computer, round.confirmed];
        banner.textContent = winner === "tie" ? `Both chose ${cap(round.confirmed)}. It's a draw!`
          : `${cap(win)} beats ${lose}. ${winner === "p1" ? player : "Computer"} wins!`;
      } else if (calibrating) banner.textContent = `Calibration read ${calibration.completed + 1} of ${calibration.total}`;
      else if (confirmed) banner.textContent = "Both shapes are locked. Press Battle!";
      else banner.textContent = phase === "review" ? "Review your EEG prediction." : "Imagine your next move.";
      $("btnBattle").classList.toggle("hidden", !confirmed);
      $("btnBattle").disabled = !!busy || !confirmed;
      $("btnRematch").classList.toggle("hidden", !revealed);
      $("btnRematch").disabled = !!busy || !revealed;
    }

    async function run(action, request) {
      if (busy) return;
      const previous = state;
      busy = action;
      capturePhase = "starting";
      error = setupError = "";
      render();
      try {
        const next = await request();
        if (action !== "start" && previous && next?.id !== previous.id) {
          throw new Error("The local EEG bridge returned another session.");
        }
        accept(next);
        if (action === "start" || action === "resume") {
          savedSession = state.id;
          try { localStorage.setItem(storageKey, savedSession); } catch { /* storage unavailable */ }
          newPlayer.value = state.player;
          setup.open = false;
        }
      } catch (err) {
        if (action === "start" || action === "resume") {
          setupError = err.message;
          if (action === "resume" && [404, 410].includes(err.status)) {
            savedSession = null;
            try { localStorage.removeItem(storageKey); } catch { /* storage unavailable */ }
          }
        } else {
          error = err.message;
          // A lost response or another tab may already have advanced this session.
          try {
            const latest = await EEG.getSession(previous.id);
            if (latest?.id === previous.id) {
              accept(latest);
              if (state.phase !== previous.phase || state.calibration.completed !== previous.calibration.completed ||
                  state.round?.id !== previous.round?.id) error = "";
            }
          } catch { /* leave the original action available for retry */ }
        }
      } finally {
        busy = null;
        render();
      }
    }

    function capture() {
      if (busy || !state || !["calibration", "ready", "result"].includes(state.phase)) return;
      const session = state.id;
      const question = state.phase === "calibration" ? state.calibration.completed : null;
      return run("capture", async () => {
        // Paint the neutral screen before the bridge starts recording.
        await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        return EEG.sessionCapture(session, question, (phase) => {
          if (busy === "capture" && state.id === session) { capturePhase = phase; render(); }
        });
      });
    }

    newGame.onclick = () => run("start", () =>
      EEG.sessionAction("session", { player: newPlayer.value.trim() || "Player" }));
    resume.onclick = () => {
      if (savedSession) return run("resume", () => EEG.getSession(savedSession));
    };
    download.onclick = async () => {
      if (!browser || !state || busy || exporting) return;
      exporting = download.disabled = true;
      const message = $("sessionFileError");
      message.textContent = "";
      try {
        await EEG.downloadSession(state.id);
      } catch (err) {
        message.textContent = err.message;
      } finally {
        exporting = false;
        download.disabled = !!busy;
      }
    };
    $("stage").addEventListener("click", (event) => {
      const button = event.target.closest("[data-act], [data-pick]");
      if (!button || busy || !state) return;
      if (button.dataset.act === "capture") return capture();
      if (state.phase !== "review") return;
      if (S.LIST.includes(button.dataset.pick)) { pick = button.dataset.pick; render(); }
      const context = { session: state.id, id: state.round.id };
      if (button.dataset.act === "confirm" && pick) {
        return run("confirm", () => EEG.sessionAction("confirm", { ...context, shape: pick }));
      }
      if (button.dataset.act === "discard") return run("discard", () => EEG.sessionAction("discard", context));
    });
    $("btnBattle").onclick = () => {
      if (busy || state?.phase !== "confirmed") return;
      return run("battle", () => EEG.sessionAction("battle", { session: state.id, id: state.round.id }));
    };
    $("btnRematch").onclick = () => { if (state?.phase === "result") return capture(); };

    render();
  }

  window.NPSessionGame = { start };
})();
