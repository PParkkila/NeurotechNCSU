(async function () {
  const $ = (id) => document.getElementById(id);
  const { EVENT_NAME, BEATS } = window.NEUROPAWN_CONFIG;
  const S = window.NPShapes;
  $("eventName").textContent = EVENT_NAME;
  $("rules").innerHTML = Object.entries(BEATS).map(([winner, loser]) =>
    `<div class="rule">${S.svg(winner, 40)}<span>${winner[0].toUpperCase() + winner.slice(1)} beats ${loser}</span>${S.svg(loser, 40)}</div>`).join("");

  if (window.NPInput.loopback || window.location.protocol === "file:") {
    $("bridgeDownload").classList.add("hidden");
    $("bridgeCommand").classList.remove("hidden");
  }
  const localLink = $("openLocalCalibration");
  localLink.classList.toggle("hidden", localLink.origin === window.location.origin);
  const bridge = await window.NPInput.bridgeStatus();
  if (bridge?.mode !== "calibration") {
    if (window.NPInput.loopback) {
      $("bridgeMessage").textContent = bridge
        ? "This bridge needs an update. Pull the latest code and restart it."
        : "The calibration bridge is unavailable at this address.";
      $("bridgeRetry").classList.remove("hidden");
      $("bridgeRetry").onclick = () => window.location.reload();
    }
    return;
  }
  window.NPSessionGame.start(bridge);
  $("bridgeSetup").classList.add("hidden");
  $("setup").classList.remove("hidden");
})();
