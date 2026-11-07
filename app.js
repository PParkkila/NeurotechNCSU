(async function () {
  const $ = (id) => document.getElementById(id);
  const cfg = window.NEUROPAWN_CONFIG, API = window.NeuroPawn, B = window.NPBattle, S = window.NPShapes;
  const SHAPES = S.LIST, EEG = window.NPInput;
  const bridge = await EEG.bridgeStatus();
  if (bridge?.mode === "calibration") {
    window.NPSessionGame.start(bridge);
    return;
  }
  const store = {
    get: (k, d) => { try { const v = localStorage.getItem(k); return v === null ? d : v; } catch { return d; } },
    set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* ignore */ } }
  };
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : "");

  let code = store.get("np_code", "");
  let role = store.get("np_role", "both");
  let blind = store.get("np_blind", "0") === "1";
  let match = null, lastSig = {}, lastStatus = null, justRevealed = false;
  let matchGeneration = 0, refreshSequence = 0, lastAppliedRefresh = 0;
  const fresh = () => ({ phase: "ready", pick: null, pred: null, err: "", confirmed: null, taught: null });
  const local = { 1: fresh(), 2: fresh() };   // per-player screen state (this device only)

  const turnOf = (m) => m.status === "revealed" ? "done" : !m.p1_locked ? 1 : !m.p2_locked ? 2 : "battle";
  const allowed = (n) => role === "both" || role === String(n);
  const isTurn = (n) => match && turnOf(match) === n && allowed(n);
  const current = (n, state, matchCode) => code === matchCode && local[n] === state && isTurn(n);

  function clearSlot(n) {
    local[n].capture?.abort();
    local[n] = fresh();
  }

  function clearInactive() {
    [1, 2].forEach((n) => {
      const L = local[n];
      if (!isTurn(n) && (L.phase !== "ready" || L.pick || L.err)) clearSlot(n);
    });
  }

  // ---------------- slot rendering ----------------
  const back = (icon, text, cls) => `<div class="back ${cls || ""}"><div>${icon}</div><small>${esc(text)}</small></div>`;
  const progress = (s) => `<div class="progress"><i style="animation-duration:${s}s"></i></div>`;
  const shapeButtons = (n, sel, disabled = false) => `<div class="picker">${SHAPES.map((s) =>
    `<button class="shapeBtn ${s === sel ? "sel" : ""}" data-pick="${s}" data-slot="${n}" aria-label="${s}" ${disabled ? "disabled" : ""}>${S.svg(s, 52)}<span>${cap(s)}</span></button>`
  ).join("")}</div>`;

  function pickerHTML(n) {
    const L = local[n], err = L.err ? `<p class="err">${esc(L.err)}</p>` : "";
    if (L.phase === "starting") return `<div class="big">…</div><p>Starting read…</p>`;
    if (L.phase === "rest") return `<div class="big">+</div><p>Rest. Keep your eyes on the cross.</p>${progress(bridge?.rest_seconds ?? cfg.REST_SECONDS)}`;
    if (L.phase === "go") return `<div class="big go">GO</div><p>Imagine your shape. Hold still.</p>${progress(bridge?.imagery_seconds ?? cfg.IMAGERY_SECONDS)}`;
    if (L.phase === "decoding") return `<div class="big">…</div><p>Decoding…</p>`;
    if (L.phase === "teaching") return `<div class="big">…</div><p>Saving feedback…</p>`;
    if (L.phase === "locking") return `<div class="big">…</div><p>Locking in…</p>`;
    if (L.phase === "review") {
      const pct = L.pred.score != null ? ` (${Math.round(L.pred.score * 100)}%)` : "";
      const saved = L.taught !== null, confirmed = L.confirmed !== null;
      const model = L.pred.model === "player" ? "Personalized" : "Seed";
      const source = L.pred.id ? "EEG" : "Demo";
      const note = saved
        ? `<p class="muted tiny">Feedback saved · ${L.taught} profile examples</p>`
        : L.pred.id ? `<p class="muted tiny">${model} model · uncalibrated confidence · ${L.pred.examples} profile examples</p>` : "";
      const prompt = saved ? "Lock the confirmed shape." : "Confirm or change it.";
      const button = saved ? "Retry lock" : confirmed ? "Retry confirm" : "Confirm &amp; lock";
      const redo = saved ? "" : `<button data-act="redo" data-slot="${n}">Redo</button>`;
      return `<div class="face">${S.svg(L.pick, 100)}</div><p>${source} predicts <b>${cap(L.pred.shape)}</b>${pct}. ${prompt}</p>` +
        note + shapeButtons(n, L.pick, confirmed) +
        `<div class="row c"><button class="primary" data-act="lock" data-slot="${n}">${button}</button>${redo}</div>` + err;
    }
    const start = `<button class="primary" data-act="decode" data-slot="${n}">Start brainwave read</button>`;
    const read = bridge ? start
      : cfg.EEG_SOURCE === "headset"
        ? `<p class="muted">Waiting for the headset. Send this player's shape from the Python game (slot ${n}).</p>`
        : start + `<p class="muted tiny">Demo decode: picks a random shape until your decoder is connected.</p>`;
    return read + `<div class="or">or pick manually</div>` + shapeButtons(n, L.pick) +
      `<button data-act="lock" data-slot="${n}" ${L.pick ? "" : "disabled"}>Lock in ${L.pick ? cap(L.pick) : ""}</button>` + err;
  }

  function slotHTML(n, m, t) {
    const name = esc(m[`p${n}_name`]);
    let tag = "Waiting", body;
    if (m.status === "revealed") {
      const shape = m[`p${n}_shape`], score = m[`p${n}_score`];
      tag = m.winner === `p${n}` ? "Winner" : m.winner === "tie" ? "Draw" : "";
      body = `<div class="face ${justRevealed ? "pop" : ""}">${S.svg(shape, 150)}</div><div class="shapeLabel">${esc(cap(shape))}</div>` +
        (score != null ? `<div class="muted">EEG score ${Math.round(score * 100)}%</div>` : "");
    } else if (m[`p${n}_locked`]) {
      tag = "Locked"; body = back("🔒", "Shape locked in", "locked");
    } else if (t === n && allowed(n)) {
      tag = "Your turn"; body = pickerHTML(n);
    } else if (t === n) {
      body = back("?", `Waiting for ${m[`p${n}_name`]} on their device`);
    } else {
      body = back("?", `${m.p1_name} goes first`);
    }
    return `<div class="slotHead"><h3>${name}</h3>${tag ? `<span class="tag">${esc(tag)}</span>` : ""}</div><div class="slotBody">${body}</div>`;
  }

  // ---------------- tally / text ----------------
  function tally(m) {
    const k = "np_tally_" + code;
    let t; try { t = JSON.parse(store.get(k, "{}")); } catch { t = {}; }
    t = { p1: 0, p2: 0, tie: 0, last: null, ...t };
    if (m.status === "revealed" && m.revealed_at && t.last !== m.revealed_at) {
      t[m.winner] = (t[m.winner] || 0) + 1; t.last = m.revealed_at; store.set(k, JSON.stringify(t));
    }
    return t;
  }
  function resultText(m) {
    if (m.winner === "tie") return `Both chose ${cap(m.p1_shape)}. It's a draw!`;
    const w = m.winner, l = w === "p1" ? "p2" : "p1";
    return `${cap(m[w + "_shape"])} beats ${m[l + "_shape"]}. ${m[w + "_name"]} wins!`;
  }

  // ---------------- main render ----------------
  function render() {
    const pill = $("modePill");
    const remote = API.mode === "supabase";
    pill.textContent = bridge
      ? `● EEG bridge connected (${remote ? "Supabase" : "local matches"})`
      : remote ? "● Live (Supabase)" : "● Demo (this browser only)";
    pill.className = "pill " + (bridge || remote ? "live" : "demo");
    $("eventName").textContent = cfg.EVENT_NAME;
    const m = match;
    $("arena").classList.toggle("hidden", !m);
    $("codeShow").textContent = m ? "Match code: " + code : "";
    if (!m) return;
    const t = turnOf(m);
    [1, 2].forEach((n) => {
      const L = local[n], el = $("slot" + n);
      const sig = JSON.stringify([m.status, m[`p${n}_locked`], m.winner, t, role, L.phase, L.pick, L.pred, L.err, L.confirmed, L.taught, !!bridge, m[`p${n}_name`], m[`p${n}_shape`]]);
      if (lastSig[n] !== sig) { lastSig[n] = sig; el.innerHTML = slotHTML(n, m, t); }
      el.classList.toggle("winner", m.status === "revealed" && m.winner === "p" + n);
      el.classList.toggle("active", t === n && allowed(n));
    });
    $("turnBanner").textContent = t === "done" ? resultText(m) : t === "battle" ? "Both shapes are locked. Press Battle!" : `${m["p" + t + "_name"]}, it's your turn.`;
    $("turnBanner").classList.toggle("result", t === "done");
    $("btnBattle").classList.toggle("hidden", t === "done");
    $("btnBattle").disabled = t !== "battle";
    $("btnRematch").classList.toggle("hidden", t !== "done");
    const ty = tally(m);
    $("scoreboard").innerHTML = `<b>${esc(m.p1_name)}</b> ${ty.p1} · ${ty.tie} draw${ty.tie === 1 ? "" : "s"} · ${ty.p2} <b>${esc(m.p2_name)}</b>`;
  }

  async function renderHistory() {
    try {
      const ms = await API.listMatches();
      $("history").innerHTML = ms.length ? ms.map((m) =>
        `<li><b>${esc(m.code)}</b> ${esc(m.p1_name)} vs ${esc(m.p2_name)} · ${esc(m.status)}` +
        (m.status === "revealed" ? " · " + (m.winner === "tie" ? "draw" : esc(m[m.winner + "_name"]) + " won") : "") +
        ` <button data-open="${esc(m.code)}">Open</button></li>`).join("") : "<li class='muted'>No matches yet</li>";
    } catch (e) { /* history is optional */ }
  }

  async function refresh() {
    if (!code) return render();
    const matchCode = code, generation = matchGeneration, sequence = ++refreshSequence;
    const relevant = () => generation === matchGeneration && sequence > lastAppliedRefresh;
    try {
      const m = await API.getMatch(matchCode);
      if (!relevant()) return;
      lastAppliedRefresh = sequence;
      if (!m) { match = null; $("joinMsg").textContent = "No match with that code."; }
      else {
        justRevealed = !!match && match.status !== "revealed" && m.status === "revealed";
        match = m; $("joinMsg").textContent = "";
      }
      clearInactive();
    } catch (e) {
      if (!relevant()) return;
      lastAppliedRefresh = sequence;
      $("joinMsg").textContent = e.message;
    }
    render();
    if (match && match.status !== lastStatus) { lastStatus = match.status; renderHistory(); }
  }

  function setCode(c) {
    const generation = ++matchGeneration;
    code = (c || "").trim().toUpperCase();
    store.set("np_code", code);
    $("joinCode").value = code;
    match = null; lastSig = {}; lastStatus = null; justRevealed = false;
    clearSlot(1); clearSlot(2);
    render();
    return refresh().then(() => { if (generation === matchGeneration && match) $("setup").open = false; });
  }

  // ---------------- actions ----------------
  async function lock(n, L, auto = false) {
    const matchCode = code, phase = L.phase, shape = L.pick, pred = L.pred;
    const teach = !auto && phase === "review" && pred?.id && L.taught === null;
    if (!shape || !current(n, L, matchCode)) return;
    L.err = "";
    if (teach) L.confirmed = shape;
    L.phase = teach ? "teaching" : "locking";
    render();
    try {
      if (teach) {
        const examples = await EEG.confirm(pred.id, shape);
        if (!current(n, L, matchCode)) return;
        L.taught = examples;
        L.phase = "locking";
        render();
      }
      await API.submitShape({ code: matchCode, slot: n, shape, score: pred?.shape === shape ? pred.score : null });
      if (!current(n, L, matchCode)) return;
      // Keep the slot locked while refreshing, even if an earlier poll is still in flight.
      lastAppliedRefresh = refreshSequence;
      match = { ...match, [`p${n}_locked`]: true };
      clearSlot(n);
      render();
      await refresh();
    } catch (err) {
      if (!current(n, L, matchCode)) return;
      if (auto) clearSlot(n);
      else L.phase = phase;
      local[n].err = err.message;
      render();
    }
  }

  async function runDecode(n, L) {
    const matchCode = code, auto = blind, playerName = match[`p${n}_name`];
    const capture = new AbortController();
    L.capture = capture;
    L.phase = "starting";
    L.err = "";
    render();
    try {
      const pred = await EEG.decode(n, (phase) => {
        if (current(n, L, matchCode)) { L.phase = phase; render(); }
      }, { playerName, signal: capture.signal });
      if (!current(n, L, matchCode)) return;
      L.capture = null;
      L.pick = pred.shape;
      L.pred = pred;
      if (auto) await lock(n, L, true);
      else { L.phase = "review"; render(); }
    } catch (err) {
      if (!current(n, L, matchCode)) return;
      clearSlot(n);
      local[n].err = err.message;
      render();
    }
  }

  $("stage").addEventListener("click", async (e) => {
    const target = e.target.closest("[data-pick], [data-act]");
    if (!target) return;
    const n = +target.dataset.slot, L = local[n];
    if (!isTurn(n) || !["ready", "review"].includes(L.phase)) return;
    if (target.dataset.pick && L.confirmed === null) { L.pick = target.dataset.pick; render(); }
    if (target.dataset.act === "decode" && L.phase === "ready") await runDecode(n, L);
    if (target.dataset.act === "lock") await lock(n, L);
    if (target.dataset.act === "redo" && L.phase === "review" && L.taught === null) { clearSlot(n); render(); }
  });

  $("btnBattle").onclick = async () => {
    const matchCode = code, generation = matchGeneration;
    $("btnBattle").disabled = true;
    try { await API.battle(matchCode); } catch (e) { if (generation === matchGeneration) alert(e.message); }
    if (generation === matchGeneration) await refresh();
  };
  $("btnRematch").onclick = async () => {
    const matchCode = code, generation = matchGeneration;
    try { await API.resetMatch(matchCode); } catch (e) { if (generation === matchGeneration) alert(e.message); }
    if (generation !== matchGeneration) return;
    lastAppliedRefresh = refreshSequence;
    clearSlot(1); clearSlot(2);
    await refresh();
  };
  $("btnNew").onclick = async () => {
    const generation = matchGeneration;
    try {
      const created = await API.createMatch($("newP1").value.trim(), $("newP2").value.trim());
      if (generation === matchGeneration) await setCode(created);
    } catch (e) { if (generation === matchGeneration) $("joinMsg").textContent = e.message; }
  };
  $("btnJoin").onclick = () => setCode($("joinCode").value);
  $("history").addEventListener("click", (e) => { const b = e.target.closest("[data-open]"); if (b) setCode(b.dataset.open); });

  $("role").value = role;
  $("role").onchange = (e) => { role = e.target.value; store.set("np_role", role); clearInactive(); render(); };
  $("blind").checked = blind;
  $("blind").onchange = (e) => { blind = e.target.checked; store.set("np_blind", blind ? "1" : "0"); };

  // rules card, built from config so it always matches BEATS
  $("rules").innerHTML = Object.entries(cfg.BEATS).map(([a, b]) =>
    `<div class="rule">${S.svg(a, 40)}<span>${cap(a)} beats ${b}</span>${S.svg(b, 40)}</div>`).join("");

  window.addEventListener("storage", (e) => { if (e.key === "np_demo") refresh(); });   // other tab, same browser
  render();
  if (code) setCode(code); else renderHistory();
  setInterval(refresh, cfg.POLL_MS);
})();
