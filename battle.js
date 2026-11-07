// Battle rules (reads window.NEUROPAWN_CONFIG.BEATS).
window.NPBattle = {
  beats(a, b) { return window.NEUROPAWN_CONFIG.BEATS[a] === b; },
  decide(a, b) {
    if (a === b) return "tie";
    return this.beats(a, b) ? "p1" : "p2";
  }
};
