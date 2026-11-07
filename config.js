// ------------------------------------------------------------------
// NeuroPawn Shape Battle: settings. Everything you might want to tweak is here.
// ------------------------------------------------------------------
window.NEUROPAWN_CONFIG = {
  EVENT_NAME: "[FILL IN: event name / date]",

  // Who beats whom: "key beats value". Square beats triangle, triangle beats circle, circle beats square.
  // If you change this, also change beats() in supabase/schema.sql (only needed when using Supabase).
  BEATS: { square: "triangle", triangle: "circle", circle: "square" },

  // "demo"    = the Start button runs the rest/GO countdown, then picks a RANDOM shape (for testing the flow)
  // "headset" = the page waits for your Python decoder to send each player's shape (needs Supabase, see README)
  EEG_SOURCE: "demo",
  REST_SECONDS: 2,      // matches the 2 s rest window in the Python recorder
  IMAGERY_SECONDS: 4,   // matches the 4 s imagery window

  // Supabase (optional). Leave the placeholders to run with no backend.
  // Without it, the game works on one device (or two tabs of the same browser).
  SUPABASE_URL: "YOUR_SUPABASE_URL",
  SUPABASE_ANON_KEY: "YOUR_SUPABASE_ANON_KEY",
  POLL_MS: 2500
};
