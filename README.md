# NeurotechNCSU
Repo hosting code for Neurotech event at NCSU Fall 2026

---

**sentence desc**

Desc...

Live site:(https://neurotechncsu.vercel.app/)

This is a functioning prototype.

---

## Limitations

* if any

---

# NeuroPawn Dashboard

Static website (HTML/CSS/JS) + Supabase database, deployed on Vercel. 

## What it does
- **Dashboard**: create a match, see each player's status (shapes stay hidden), press **Battle** once both have gone, see the result and both shapes.
- **Play**: draw a shape on a canvas and lock it in (strokes are saved too, for the "drawing signals" idea). Game iframe placeholder lives here.
- **Questionnaire**: editable list in `js/questions.js`; answers saved to Supabase.
- **NeuroPawn signals**: placeholder table + `NeuroPawn.saveSignal()`; fill in once the device exists.
- Rules: square > triangle > circle > square, or "best-drawn shape" (score from the drawing heuristic in `js/draw.js`).

## Folder layout
```
index.html        page
css/styles.css    styling
js/config.js      <-- paste Supabase URL + anon key here
js/api.js         all database calls (Supabase, or demo mode)
js/app.js         UI logic
js/battle.js      battle rules
js/draw.js        drawing pad + rough shape scoring
js/questions.js   questionnaire (edit freely)
game/             <-- put the game here (see game/README.md)
supabase/schema.sql  database setup
vercel.json
```

## Setup
1. **Try it first**: open `index.html` (or run `npx serve .`). With no config it runs in *demo mode* (data stays in your browser only).
2. **Supabase**: create a project, open SQL Editor, paste and run `supabase/schema.sql`.
3. Copy **Project URL** and **anon public key** (Project Settings → API) into `js/config.js`.
4. **GitHub**: put this folder in your repo (e.g. `/dashboard`, or at the repo root).
5. **Vercel**: if it's in a subfolder, set *Project Settings → Root Directory* to that folder (or add it as a separate Vercel project). Framework preset: **Other**. Push, and it deploys.

## Notes / limits (rough draft)
- Hidden shapes are enforced in the database: the site reads from the `matches_public` view, which returns shapes only after `status = 'revealed'`.
- There are no logins. Anyone with a match code can submit or press Battle. Fine for a demo or club event; add Supabase Auth before anything public.
- The shape detection in `js/draw.js` is a simple circularity heuristic, so expect it to misjudge sloppy drawings. In the default rule the shape the player *selects* is what counts.
- Live updates use polling every 3 s (`POLL_MS` in `config.js`).

## Credits

**Team:** Ishan Hitesh Mistry, Khrystyna Lukecha, Patrik Parkkila, Natalia Bernardo

Made for Neurotech Hackathon 2026

**AI assistance:** Most of the code in this project was written with AI assistance:

* **Claude** (Anthropic) wrote the code for the game dashboard. 
* **Codex** (OpenAI) wrote the code for the overall game, well as prototyping and calibration.

The team defined the project goals, directed the design, built and wired the hardware, tested the system, and integrated and debugged everything.

**Services and tools:** NeuroPawn, Vercel
