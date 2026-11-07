// ------------------------------------------------------------------
// Where the brainwave decoder plugs in.
//
// decode(slot, onPhase) must resolve to { shape: "circle"|"triangle"|"square", score: 0..1 }.
// It may call onPhase("rest"), onPhase("go"), onPhase("decoding") so the screen shows the
// same prompts as the recorder (rest cross, GO, imagine).
//
// Default below is a DEMO: it runs the countdown and returns a RANDOM shape.
// To use real EEG, either:
//   (a) set EEG_SOURCE: "headset" in config.js and send shapes from Python (see README), or
//   (b) replace this function with a call to your own decoder service.
// ------------------------------------------------------------------
window.NPInput = {
  async decode(slot, onPhase) {
    const cfg = window.NEUROPAWN_CONFIG;
    const wait = (s) => new Promise((r) => setTimeout(r, s * 1000));
    onPhase("rest");
    await wait(cfg.REST_SECONDS);
    onPhase("go");
    await wait(cfg.IMAGERY_SECONDS);
    onPhase("decoding");
    await wait(0.6);
    const shapes = window.NPShapes.LIST;
    return { shape: shapes[Math.floor(Math.random() * shapes.length)], score: +(0.34 + Math.random() * 0.4).toFixed(2) };
  }
};
