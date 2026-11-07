// Shape artwork (same palette as the Python drawing.py).
window.NPShapes = {
  LIST: ["circle", "triangle", "square"],
  COLORS: { circle: ["#7ad3df", "#177687"], triangle: ["#c8b1f4", "#63479b"], square: ["#ffcf8a", "#945624"] },
  svg(shape, size) {
    const c = this.COLORS[shape];
    if (!c) return "";
    const g = {
      circle: '<circle cx="50" cy="50" r="40"/>',
      triangle: '<polygon points="50,10 92,86 8,86"/>',
      square: '<rect x="12" y="12" width="76" height="76" rx="3"/>'
    }[shape];
    return `<svg viewBox="0 0 100 100" width="${size}" height="${size}" role="img" aria-label="${shape}" ` +
      `fill="${c[0]}" stroke="${c[1]}" stroke-width="3" stroke-linejoin="round">${g}</svg>`;
  }
};
