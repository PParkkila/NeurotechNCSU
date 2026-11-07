(function () {
  const $ = (id) => document.getElementById(id);
  const cfg = window.NEUROPAWN_CONFIG, API = window.NeuroPawn, B = window.NPBattle, S = window.NPShapes;
  const SHAPES = S.LIST;
  const store = {
    get: (k, d) => { try { const v = localStorage.getItem(k); return v === null ? d : v; } catch { return d; } },
    set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* ignore */ } }
  };
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : "");

  let code = store.get("np_code", "");
  let role = store.get("np_role", "both");
  let blind = store.get("np_blind", "1") === "1";
  let match = null, lastSig = {}, lastStatus = null, justRevealed = false;
  const fresh = () => ({ phase: "ready", pick: null, pred: null, err: "" });
  const local = { 1: fresh(), 2: fresh() };   // per-player screen state (this device only)

  const turnOf = (m) => m.status === "revealed" ? "done" : !m.p1_locked ? 1 : !m.p2_locked ? 2 : "battle";
  const allowed = (n) => role === "both" || role === String(n);

  // ---------------- slot rendering ----------------
  const back = (icon, text, cls) => `<div class="back ${cls || ""}"><div>${icon}</div><small>${esc(text)}</small></div>`;
  const progress = (s) => `<div class="progress"><i style="animation-duration:${s}s"></i></div>`;
  const shapeButtons = (n, sel) => `<div class="picker">${SHAPES.map((s) =>
    `<button class="shapeBtn ${s === sel ? "sel" : ""}" data-pick="${s}" data-slot="${n}" aria-label="${s}">${S.svg(s, 52)}<span>${cap(s)}</span></button>`
  ).join("")}</div>`;

  function pickerHTML(n) {
    const L = local[n];
    if (L.phase === "rest") return `<div class="big">+</div><p>Rest. Keep your eyes on the cross.</p>${progress(cfg.REST_SECONDS)}`;
    if (L.phase === "go") return `<div class="big go">GO</div><p>Imagine your shape. Hold still.</p>${progress(cfg.IMAGERY_SECONDS)}`;
    if (L.phase === "decoding") return `<div class="big">…</div><p>Decoding…</p>`;
    if (L.phase === "locking") return `<div class="big">…</div><p>Locking in…</p>`;
    if (L.phase === "review") {
      const pct = L.pred.score != null ? ` (${Math.round(L.pred.score * 100)}%)` : "";
      return `<div class="face">${S.svg(L.pick, 100)}</div><p>EEG predicts <b>${cap(L.pred.shape)}</b>${pct}. Confirm or change it.</p>` +
        shapeButtons(n, L.pick) +
        `<div class="row c"><button class="primary" data-act="lock" data-slot="${n}">Confirm &amp; lock</button><button data-act="redo" data-slot="${n}">Redo</button></div>`;
    }
    const read = cfg.EEG_SOURCE === "headset"
      ? `<p class="muted">Waiting for the headset. Send this player's shape from the Python game (slot ${n}).</p>`
      : `<button class="primary" data-act="decode" data-slot="${n}">Start brainwave read</button><p class="muted tiny">Demo decode: picks a random shape until your decoder is connected.</p>`;
    return read + `<div class="or">or pick manually</div>` + shapeButtons(n, L.pick) +
      `<button data-act="lock" data-slot="${n}" ${L.pick ? "" : "disabled"}>Lock in ${L.pick ? cap(L.pick) : ""}</button>` +
      (L.err ? `<p class="err">${esc(L.err)}</p>` : "");
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
    return `<div class="slotHead"><h3>${name}</h3><span class="tag">${esc(tag)}</span></div><div class="slotBody">${body}</div>`;
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
    pill.textContent = API.mode === "supabase" ? "● Live (Supabase)" : "● Demo (this browser only)";
    pill.className = "pill " + (API.mode === "supabase" ? "live" : "demo");
    $("eventName").textContent = cfg.EVENT_NAME;
    const m = match;
    $("arena").classList.toggle("hidden", !m);
    $("codeShow").textContent = m ? "Match code: " + code : "";
    if (!m) return;
    const t = turnOf(m);
    [1, 2].forEach((n) => {
      const L = local[n], el = $("slot" + n);
      const sig = JSON.stringify([m.status, m[`p${n}_locked`], m.winner, t, role, L.phase, L.pick, L.err, m[`p${n}_name`], m[`p${n}_shape`]]);
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
    try {
      const m = await API.getMatch(code);
      if (!m) { match = null; $("joinMsg").textContent = "No match with that code."; }
      else {
        justRevealed = !!match && match.status !== "revealed" && m.status === "revealed";
        match = m; $("joinMsg").textContent = "";
      }
    } catch (e) { $("joinMsg").textContent = e.message; }
    render();
    if (match && match.status !== lastStatus) { lastStatus = match.status; renderHistory(); }
  }

  function setCode(c) {
    code = (c || "").trim().toUpperCase();
    store.set("np_code", code);
    $("joinCode").value = code;
    match = null; lastSig = {}; lastStatus = null; local[1] = fresh(); local[2] = fresh();
    return refresh().then(() => { if (match) $("setup").open = false; });
  }

  // ---------------- actions ----------------
  async function lock(n, shape, score) {
    if (!shape) return;
    local[n].phase = "locking"; render();
    await API.submitShape({ code, slot: n, shape, score });
    local[n] = fresh(); await refresh();
  }
  async function runDecode(n) {
    const r = await window.NPInput.decode(n, (ph) => { local[n].phase = ph; render(); });
    if (blind) return lock(n, r.shape, r.score);
    local[n] = { phase: "review", pick: r.shape, pred: r, err: "" }; render();
  }

  $("stage").addEventListener("click", async (e) => {
    const pick = e.target.closest("[data-pick]");
    if (pick) { local[+pick.dataset.slot].pick = pick.dataset.pick; render(); return; }
    const act = e.target.closest("[data-act]");
    if (!act) return;
    const n = +act.dataset.slot, L = local[n];
    try {
      if (act.dataset.act === "decode") await runDecode(n);
      else if (act.dataset.act === "lock") await lock(n, L.pick, L.pred && L.pred.shape === L.pick ? L.pred.score : null);
      else if (act.dataset.act === "redo") { local[n] = fresh(); render(); }
    } catch (err) { local[n] = { ...fresh(), err: err.message }; render(); }
  });

  $("btnBattle").onclick = async () => {
    $("btnBattle").disabled = true;
    try { await API.battle(code); } catch (e) { alert(e.message); }
    await refresh();
  };
  $("btnRematch").onclick = async () => {
    try { await API.resetMatch(code); } catch (e) { alert(e.message); }
    local[1] = fresh(); local[2] = fresh(); await refresh();
  };
  $("btnNew").onclick = async () => {
    try { await setCode(await API.createMatch($("newP1").value.trim(), $("newP2").value.trim())); }
    catch (e) { $("joinMsg").textContent = e.message; }
  };
  $("btnJoin").onclick = () => setCode($("joinCode").value);
  $("history").addEventListener("click", (e) => { const b = e.target.closest("[data-open]"); if (b) setCode(b.dataset.open); });

  $("role").value = role;
  $("role").onchange = (e) => { role = e.target.value; store.set("np_role", role); render(); };
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
