"""Render visual imagery predictions as standalone SVGs."""

from html import escape
from pathlib import Path


LAYOUT_NAMES = ("Circle above square", "Square above circle")
SHAPE_NAMES = ("Circle", "Square")
_SHAPES = (("circle", "square"), ("square", "circle"))


def draw_layout(
    path: str | Path,
    prediction: int,
    *,
    actual: int | None = None,
    confidence: float | None = None,
    trial: int | None = None,
    subtitle: str | None = None,
) -> Path:
    """Draw class 0 (circle first) or class 1 (square first) from top to bottom."""
    if prediction not in (0, 1) or (actual is not None and actual not in (0, 1)):
        raise ValueError("Layouts must be class 0 or class 1.")
    if confidence is not None and not 0 <= confidence <= 1:
        raise ValueError("Model score must be between 0 and 1.")

    name = LAYOUT_NAMES[prediction]
    details = []
    if trial is not None:
        details.append(f"Trial {trial}")
    if confidence is not None:
        details.append(f"Model score {confidence:.0%}")
    metadata = escape("  /  ".join(details))
    subtitle = escape(subtitle or "NeuroPawn visual imagery prototype")

    rows = []
    for y, label, shape in zip((275, 420), ("TOP", "BOTTOM"), _SHAPES[prediction]):
        if shape == "circle":
            geometry = (
                f'<circle cx="380" cy="{y}" r="43" '
                'fill="#7ad3df" stroke="#177687" stroke-width="3"/>'
            )
        else:
            geometry = (
                f'<rect x="340" y="{y - 40}" width="80" height="80" rx="4" '
                'fill="#ffcf8a" stroke="#945624" stroke-width="3"/>'
            )
        rows.append(
            f'<text class="label" x="90" y="{y + 5}">{label}</text>'
            f'{geometry}<text class="body" x="585" y="{y + 6}">{shape.title()}</text>'
        )

    comparison = "Target hidden"
    if actual is not None:
        result = "Match" if prediction == actual else "Different"
        comparison = f"{result}  /  Target was {LAYOUT_NAMES[actual].lower()}"

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 640"
     width="760" height="640" role="img" aria-labelledby="title description">
  <title id="title">Predicted layout. {name}</title>
  <desc id="description">{name}. {comparison}.</desc>
  <style>
    text {{ font-family: system-ui, sans-serif; fill: #172b3a; }}
    .label {{ font-size: 13px; font-weight: 700; letter-spacing: 2px; fill: #4b6572; }}
    .body {{ font-size: 17px; }}
  </style>
  <rect width="760" height="640" fill="#eaf0f3"/>
  <rect x="28" y="28" width="704" height="584" rx="20" fill="#fff" stroke="#d0dce3"/>
  <text class="label" x="58" y="73">PREDICTED LAYOUT</text>
  <text x="58" y="122" font-size="38" font-weight="700">{name}</text>
  <text class="body" x="58" y="156">{subtitle}</text>
  <rect x="58" y="201" width="644" height="292" rx="14" fill="#f4f8fa" stroke="#d0dce3"/>
  <line x1="80" y1="347" x2="680" y2="347" stroke="#d0dce3" stroke-dasharray="4 6"/>
  {"".join(rows)}
  <text x="58" y="545" font-size="19" font-weight="600">{comparison}</text>
  <text class="body" x="58" y="576">{metadata}</text>
</svg>
"""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    return output


def draw_shape(
    path: str | Path,
    prediction: int,
    *,
    actual: int | None = None,
    confidence: float | None = None,
    trial: int | None = None,
    subtitle: str | None = None,
    names: tuple[str, ...] = SHAPE_NAMES,
) -> Path:
    """Draw a predicted shape using the recording's class order."""
    if prediction not in range(len(names)) or (actual is not None and actual not in range(len(names))):
        raise ValueError("Shape index is outside the recording's classes.")
    if confidence is not None and not 0 <= confidence <= 1:
        raise ValueError("Model score must be between 0 and 1.")

    name = names[prediction]
    geometry = {
        "Circle": '<circle cx="380" cy="347" r="90" fill="#7ad3df" stroke="#177687" stroke-width="3"/>',
        "Triangle": '<polygon points="380,257 470,437 290,437" fill="#c8b1f4" '
                    'stroke="#63479b" stroke-width="3"/>',
        "Square": '<rect x="290" y="257" width="180" height="180" fill="#ffcf8a" '
                  'stroke="#945624" stroke-width="3"/>',
    }[name]
    details = []
    if trial is not None:
        details.append(f"Trial {trial}")
    if confidence is not None:
        details.append(f"Model score {confidence:.0%}")
    metadata = escape("  /  ".join(details))
    subtitle = escape(subtitle or "NeuroPawn visual imagery prototype")

    comparison = "Target hidden"
    if actual is not None:
        result = "Match" if prediction == actual else "Different"
        comparison = f"{result}  /  Target was {names[actual].lower()}"

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 640"
     width="760" height="640" role="img" aria-labelledby="title description">
  <title id="title">Predicted shape. {name}</title>
  <desc id="description">{name}. {comparison}.</desc>
  <style>
    text {{ font-family: system-ui, sans-serif; fill: #172b3a; }}
    .label {{ font-size: 13px; font-weight: 700; letter-spacing: 2px; fill: #4b6572; }}
    .body {{ font-size: 17px; }}
  </style>
  <rect width="760" height="640" fill="#eaf0f3"/>
  <rect x="28" y="28" width="704" height="584" rx="20" fill="#fff" stroke="#d0dce3"/>
  <text class="label" x="58" y="73">PREDICTED SHAPE</text>
  <text x="58" y="122" font-size="38" font-weight="700">{name}</text>
  <text class="body" x="58" y="156">{subtitle}</text>
  <rect x="58" y="201" width="644" height="292" rx="14" fill="#f4f8fa" stroke="#d0dce3"/>
  {geometry}
  <text x="58" y="545" font-size="19" font-weight="600">{comparison}</text>
  <text class="body" x="58" y="576">{metadata}</text>
</svg>
"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    return output
