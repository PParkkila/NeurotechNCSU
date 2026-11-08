// Use the same-origin EEG bridge when the game is served locally.
(function () {
  const loopback = ["localhost", "127.0.0.1", "[::1]"].includes(window.location.hostname);
  async function bridgeStatus() {
    if (!loopback) return null;
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

  async function post(path, data) {
    return checked(await fetch(path, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data)
    }));
  }

  async function streamCapture(data, onPhase) {
    const response = await post("/eeg/capture", data);
    if (!response.body) throw new Error("The local EEG bridge did not start a read.");
    const reader = response.body.getReader(), text = new TextDecoder();
    let pending = "", result;
    function consume(line) {
      if (!line.trim()) return;
      const event = JSON.parse(line);
      if (typeof event.error === "string") throw new Error(event.error);
      if (["rest", "go", "decoding"].includes(event.phase)) onPhase(event.phase);
      if (event.state) result = event.state;
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

  window.NPInput = {
    loopback,
    bridgeStatus,
    async getSession(session) {
      const response = await checked(await fetch("/eeg/session?session=" + encodeURIComponent(session), { cache: "no-store" }));
      return response.json();
    },
    async sessionAction(action, data) {
      return (await post("/eeg/" + action, data)).json();
    },
    sessionCapture(session, question, onPhase) {
      return streamCapture({ session, question }, onPhase);
    }
  };
})();
