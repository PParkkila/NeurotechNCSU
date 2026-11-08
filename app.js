(async function () {
  const $ = (id) => document.getElementById(id);
  const { EVENT_NAME, BEATS } = window.NEUROPAWN_CONFIG;
  const S = window.NPShapes;
  $("eventName").textContent = EVENT_NAME;
  $("rules").innerHTML = Object.entries(BEATS).map(([winner, loser]) =>
    `<div class="rule">${S.svg(winner, 40)}<span>${winner[0].toUpperCase() + winner.slice(1)} beats ${loser}</span>${S.svg(loser, 40)}</div>`).join("");

  const bridge = await window.NPInput.bridgeStatus();
  if (bridge?.mode === "calibration") {
    window.NPSessionGame.start(bridge, window.NPInput);
    $("bridgeSetup").classList.add("hidden");
    $("setup").classList.remove("hidden");
    return;
  }

  const connect = $("btnConnect"), disconnect = $("btnDisconnect"), message = $("bridgeMessage");
  if (window.NPInput.loopback || window.location.protocol === "file:") {
    $("bridgeDownload").classList.add("hidden");
    $("bridgeCommand").classList.remove("hidden");
  }
  const localLink = $("openLocalCalibration");
  localLink.classList.toggle("hidden", localLink.origin === window.location.origin);
  if (!window.isSecureContext || !navigator.serial || !window.NPSerial) {
    $("bridgeFallback").open = true;
    message.textContent = "Open this HTTPS page in desktop Chrome to connect over USB.";
    return;
  }

  const game = window.NPBrowserGame.create();
  let device = null, started = false;
  connect.disabled = false;

  function offline(text) {
    device = null;
    game.setBoard(null);
    disconnect.classList.add("hidden");
    $("bridgeSetup").classList.remove("hidden");
    $("modePill").textContent = "EEG headset disconnected";
    $("modePill").className = "pill demo";
    message.textContent = text;
  }

  connect.onclick = async () => {
    connect.disabled = true;
    try {
      const inputs = $("boardInputs").value.split(",").map(Number);
      if (!inputs.length || inputs.some((input) => !Number.isInteger(input) || input < 1 || input > 8) ||
          new Set(inputs).size !== inputs.length) {
        throw new Error("Enter unique board input numbers from 1 to 8, separated by commas.");
      }
      message.textContent = "Choose your NeuroPawn in the browser’s USB serial picker.";
      device = await window.NPSerial.connect({
        inputs, gain: Number($("boardGain").value),
        onStatus: (text) => { message.textContent = text; },
        onDisconnect: () => offline("NeuroPawn disconnected. Plug it in and click Connect again."),
      });
      game.setBoard(device);
      if (!started) {
        window.NPSessionGame.start(game.bridge, game);
        started = true;
      }
      $("modePill").textContent = "● NeuroPawn connected over USB";
      $("modePill").className = "pill live";
      disconnect.classList.remove("hidden");
      $("bridgeSetup").classList.add("hidden");
      $("setup").classList.remove("hidden");
    } catch (error) {
      const problem = error.message || "Could not connect to NeuroPawn. Close other headset apps and try again.";
      if (device) {
        const previous = device;
        offline(problem);
        await previous.close().catch(() => {});
      } else {
        message.textContent = problem;
      }
    } finally {
      connect.disabled = false;
    }
  };

  disconnect.onclick = async () => {
    const previous = device;
    connect.disabled = true;
    offline("Connect NeuroPawn again to continue recording.");
    try {
      await previous?.close();
    } catch (error) {
      message.textContent = error.message;
    } finally {
      connect.disabled = false;
    }
  };
  window.addEventListener("pagehide", () => {
    if (device) {
      const previous = device;
      offline("Connect NeuroPawn again to continue recording.");
      void previous.close().catch(() => {});
    }
  });
})();
