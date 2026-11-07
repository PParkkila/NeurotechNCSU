// Use the same-origin EEG bridge when the game is served locally.
(function () {
  const loopback = ["localhost", "127.0.0.1", "[::1]"].includes(window.location.hostname);
  const profiles = new Map();
  let statusRequest;

  function bridgeStatus() {
    if (!loopback) return Promise.resolve(null);
    statusRequest ??= (async () => {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 1500);
      try {
        const response = await fetch("/eeg/status", { cache: "no-store", signal: controller.signal });
        if (!response.ok) return null;
        const status = await response.json();
        if (status.available !== true || !Number.isFinite(status.rest_seconds) || status.rest_seconds < 0 ||
            !Number.isFinite(status.imagery_seconds) || status.imagery_seconds <= 0) return null;
        return status;
      } catch {
        return null;
      } finally {
        clearTimeout(timeout);
      }
    })();
    return statusRequest;
  }

  function profileFor(slot, playerName) {
    const key = "np_eeg_profile_" + JSON.stringify([slot, String(playerName ?? "").normalize("NFC").trim()]);
    if (profiles.has(key)) return profiles.get(key);
    let id;
    try { id = localStorage.getItem(key); } catch { /* unavailable */ }
    if (!/^[\da-f]{8}(?:-[\da-f]{4}){3}-[\da-f]{12}$/i.test(id ?? "")) {
      id = window.crypto.randomUUID();
      try { localStorage.setItem(key, id); } catch { /* retain for this tab */ }
    }
    profiles.set(key, id);
    return id;
  }

  async function checked(response) {
    if (!response.ok) {
      const detail = await response.json().catch(() => null);
      const error = new Error(typeof detail?.error === "string" ? detail.error : "The local EEG bridge could not complete the request.");
      error.status = response.status;
      throw error;
    }
    return response;
  }

  async function post(path, data, signal) {
    return checked(await fetch(path, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data), signal
    }));
  }

  async function streamCapture(data, resultField, onPhase, signal) {
    const response = await post("/eeg/capture", data, signal);
    if (!response.body) throw new Error("The local EEG bridge did not start a read.");
    const reader = response.body.getReader(), text = new TextDecoder();
    let pending = "", result;
    function consume(line) {
      if (!line.trim()) return;
      const event = JSON.parse(line);
      if (typeof event.error === "string") throw new Error(event.error);
      if (["rest", "go", "decoding"].includes(event.phase)) onPhase(event.phase);
      if (event[resultField]) result = event[resultField];
    }
    try {
      while (!result) {
        const { value, done } = await reader.read();
        pending += text.decode(value, { stream: !done });
        const lines = pending.split("\n");
        pending = lines.pop();
        for (const line of lines) {
          consume(line);
          if (result) break;
        }
        if (done) {
          if (!result) consume(pending);
          break;
        }
      }
    } finally {
      await reader.cancel().catch(() => {});
      reader.releaseLock();
    }
    if (!result) throw new Error("The headset read ended without a result.");
    return result;
  }

  async function capture(slot, playerName, onPhase, signal) {
    const p = await streamCapture({ profile: profileFor(slot, playerName) }, "prediction", onPhase, signal);
    if (typeof p.id !== "string" || !p.id || !window.NPShapes.LIST.includes(p.shape) ||
        !Number.isFinite(p.score) || p.score < 0 || p.score > 1 ||
        !["seed", "player"].includes(p.model) || !Number.isInteger(p.examples) || p.examples < 0) {
      throw new Error("The local EEG bridge returned an invalid prediction.");
    }
    return p;
  }

  async function demo(onPhase) {
    const cfg = window.NEUROPAWN_CONFIG;
    const wait = (s) => new Promise((resolve) => setTimeout(resolve, s * 1000));
    onPhase("rest");
    await wait(cfg.REST_SECONDS);
    onPhase("go");
    await wait(cfg.IMAGERY_SECONDS);
    onPhase("decoding");
    await wait(0.6);
    const shapes = window.NPShapes.LIST;
    return { shape: shapes[Math.floor(Math.random() * shapes.length)], score: +(0.34 + Math.random() * 0.4).toFixed(2) };
  }

  window.NPInput = {
    bridgeStatus,
    async decode(slot, onPhase, { playerName, signal } = {}) {
      return await bridgeStatus() ? capture(slot, playerName, onPhase, signal) : demo(onPhase);
    },
    async confirm(id, shape) {
      const response = await post("/eeg/confirm", { id, shape });
      const { examples } = await response.json();
      if (!Number.isInteger(examples) || examples < 0) throw new Error("The local EEG bridge did not acknowledge feedback.");
      return examples;
    },
    async getSession(session) {
      const response = await checked(await fetch("/eeg/session?session=" + encodeURIComponent(session), { cache: "no-store" }));
      return response.json();
    },
    async sessionAction(action, data) {
      return (await post("/eeg/" + action, data)).json();
    },
    sessionCapture(session, question, onPhase) {
      return streamCapture({ session, question }, "state", onPhase);
    }
  };
})();
