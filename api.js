// Data layer. Supabase when configured, otherwise a browser-only demo store.
(function () {
  const cfg = window.NEUROPAWN_CONFIG;
  const configured = cfg.SUPABASE_URL && !cfg.SUPABASE_URL.startsWith("YOUR_") &&
                     cfg.SUPABASE_ANON_KEY && !cfg.SUPABASE_ANON_KEY.startsWith("YOUR_") && window.supabase;
  const sb = configured ? window.supabase.createClient(cfg.SUPABASE_URL, cfg.SUPABASE_ANON_KEY) : null;
  const SHAPES = window.NPShapes.LIST;

  // ---------- demo backend (localStorage; shared between tabs of one browser) ----------
  const read = () => { try { return JSON.parse(localStorage.getItem("np_demo") || "{}"); } catch { return {}; } };
  const write = (d) => localStorage.setItem("np_demo", JSON.stringify(d));
  const view = (m) => {
    const rev = m.status === "revealed";
    return {
      code: m.code, p1_name: m.p1_name, p2_name: m.p2_name, status: m.status, winner: m.winner,
      created_at: m.created_at, revealed_at: m.revealed_at,
      p1_locked: !!m.p1_shape, p2_locked: !!m.p2_shape,
      p1_shape: rev ? m.p1_shape : null, p1_score: rev ? m.p1_score : null,
      p2_shape: rev ? m.p2_shape : null, p2_score: rev ? m.p2_score : null
    };
  };
  const demo = {
    async createMatch(p1, p2) {
      const d = read(); const code = Math.random().toString(36).slice(2, 7).toUpperCase();
      d[code] = { code, p1_name: p1 || "Player 1", p2_name: p2 || "Player 2", status: "waiting", created_at: new Date().toISOString() };
      write(d); return code;
    },
    async getMatch(code) { const m = read()[code]; return m ? view(m) : null; },
    async listMatches() {
      return Object.values(read()).sort((a, b) => b.created_at.localeCompare(a.created_at)).slice(0, 10).map(view);
    },
    async submitShape({ code, slot, shape, score }) {
      const d = read(), m = d[code];
      if (!m) throw new Error("Match not found");
      if (m.status === "revealed") throw new Error("Match already revealed");
      if (!SHAPES.includes(shape)) throw new Error("Bad shape");
      m[`p${slot}_shape`] = shape; m[`p${slot}_score`] = score ?? null;
      if (m.p1_shape && m.p2_shape) m.status = "ready";
      write(d);
    },
    async battle(code) {
      const d = read(), m = d[code];
      if (!m || !m.p1_shape || !m.p2_shape) throw new Error("Both players must lock in first");
      if (m.status === "revealed") return;
      Object.assign(m, { status: "revealed", winner: window.NPBattle.decide(m.p1_shape, m.p2_shape), revealed_at: new Date().toISOString() });
      write(d);
    },
    async resetMatch(code) {
      const d = read(), m = d[code]; if (!m) return;
      ["p1_shape", "p1_score", "p2_shape", "p2_score", "winner", "revealed_at"].forEach((k) => delete m[k]);
      m.status = "waiting"; write(d);
    }
  };

  // ---------- Supabase backend ----------
  const ok = ({ data, error }) => { if (error) throw new Error(error.message); return data; };
  const remote = {
    async createMatch(p1, p2) { return ok(await sb.rpc("create_match", { p_p1: p1, p_p2: p2 })); },
    async getMatch(code) { return ok(await sb.from("matches_public").select("*").eq("code", code).maybeSingle()); },
    async listMatches() { return ok(await sb.from("matches_public").select("*").order("created_at", { ascending: false }).limit(10)); },
    async submitShape({ code, slot, shape, score }) {
      ok(await sb.rpc("submit_shape", { p_code: code, p_slot: slot, p_shape: shape, p_score: score ?? null }));
    },
    async battle(code) { ok(await sb.rpc("battle_match", { p_code: code })); },
    async resetMatch(code) { ok(await sb.rpc("reset_match", { p_code: code })); }
  };

  window.NeuroPawn = { mode: configured ? "supabase" : "demo", ...(configured ? remote : demo) };
})();
