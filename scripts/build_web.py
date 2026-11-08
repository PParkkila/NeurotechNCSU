"""Build the static Shape Battle page and its downloadable local bridge."""

from pathlib import Path
from shutil import copyfile, rmtree
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "web-dist"
WEB_FILES = (
    "index.html", "config.js", "shapes.js", "battle.js", "input.js", "session-game.js", "app.js",
    "browser-serial.js", "browser-decoder.js", "browser-confidence.js", "browser-store.js", "browser-game.js",
)
BRIDGE_FILES = (
    *WEB_FILES,
    "LICENSE",
    "requirements.txt",
    "imagery/__init__.py",
    "imagery/__main__.py",
    "imagery/drawing.py",
    "imagery/model.py",
    "imagery/personalization.py",
    "imagery/recording.py",
    "imagery/rps.py",
    "imagery/selection.html",
    "imagery/selection.py",
    "imagery/ssvep.py",
    "imagery/web.py",
    "imagery/web_session.py",
)


def main():
    if DIST.exists():
        rmtree(DIST)
    DIST.mkdir()
    for name in WEB_FILES:
        copyfile(ROOT / name, DIST / name)
    with ZipFile(DIST / "neuropawn-local.zip", "w", ZIP_DEFLATED) as archive:
        for name in BRIDGE_FILES:
            archive.write(ROOT / name, f"neuropawn-local/{name}")
        archive.write(ROOT / "docs/local-bridge.md", "neuropawn-local/README.md")
    print(f"Drop {DIST} into Vercel Drop.")


if __name__ == "__main__":
    main()
