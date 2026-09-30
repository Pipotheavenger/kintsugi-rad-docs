// Renders the code-map diagrams (.kmap) at natural size, with clickable boxes,
// and adds a full-screen viewer (zoom with wheel/buttons, drag to pan, Esc to close).
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";

const dark = document.body.getAttribute("data-md-color-scheme") === "slate";
mermaid.initialize({
  startOnLoad: false,
  securityLevel: "loose", // needed for `click` links to the code
  theme: dark ? "dark" : "neutral",
  themeVariables: { fontSize: "15px" },
  flowchart: { useMaxWidth: false, htmlLabels: true, wrappingWidth: 340, nodeSpacing: 28, rankSpacing: 48 },
});
await mermaid.run({ querySelector: ".kmap" });

function titleFor(el) {
  // Nearest preceding heading gives the viewer its title.
  for (let n = el.previousElementSibling; n; n = n.previousElementSibling) {
    if (/^H[1-4]$/.test(n.tagName)) return n.textContent.replace("¶", "").trim();
  }
  return document.title;
}

function openViewer(el) {
  const original = el.querySelector("svg");
  const { width: w, height: h } = original.getBoundingClientRect(); // natural size
  const svg = original.cloneNode(true);
  svg.removeAttribute("style");
  svg.setAttribute("width", w);
  svg.setAttribute("height", h);

  const overlay = document.createElement("div");
  overlay.className = "kmap-full";
  overlay.innerHTML = `
    <div class="kmap-bar">
      <strong>${titleFor(el)}</strong>
      <span class="kmap-hint">scroll = zoom · drag = move · click a box = open code · Esc = close</span>
      <span class="kmap-tools">
        <button data-a="out" title="Zoom out">−</button>
        <button data-a="in" title="Zoom in">+</button>
        <button data-a="fit" title="Fit to screen">Fit</button>
        <button data-a="one" title="Actual size">1:1</button>
        <button data-a="close" title="Close (Esc)">✕</button>
      </span>
    </div>
    <div class="kmap-stage"><div class="kmap-canvas"></div></div>`;
  const stage = overlay.querySelector(".kmap-stage");
  const canvas = overlay.querySelector(".kmap-canvas");
  canvas.appendChild(svg);
  document.body.appendChild(overlay);
  document.documentElement.style.overflow = "hidden";

  let scale = 1, x = 0, y = 0;
  const apply = () => (canvas.style.transform = `translate(${x}px, ${y}px) scale(${scale})`);
  const fit = () => {
    const r = stage.getBoundingClientRect();
    scale = Math.min(r.width / w, r.height / h) * 0.96;
    x = (r.width - w * scale) / 2;
    y = (r.height - h * scale) / 2;
    apply();
  };
  const zoomAt = (factor, cx, cy) => {
    const next = Math.min(4, Math.max(0.1, scale * factor));
    x = cx - (cx - x) * (next / scale);
    y = cy - (cy - y) * (next / scale);
    scale = next;
    apply();
  };
  const center = () => {
    const r = stage.getBoundingClientRect();
    return [r.width / 2, r.height / 2];
  };

  stage.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = stage.getBoundingClientRect();
    zoomAt(e.deltaY < 0 ? 1.15 : 1 / 1.15, e.clientX - r.left, e.clientY - r.top);
  }, { passive: false });

  let drag = null, moved = false;
  stage.addEventListener("pointerdown", (e) => {
    drag = { sx: e.clientX, sy: e.clientY, x, y };
    moved = false;
  });
  window.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.sx, dy = e.clientY - drag.sy;
    if (Math.abs(dx) + Math.abs(dy) > 4) moved = true;
    if (moved) {
      x = drag.x + dx;
      y = drag.y + dy;
      apply();
      stage.classList.add("dragging");
    }
  });
  window.addEventListener("pointerup", () => {
    drag = null;
    stage.classList.remove("dragging");
  });
  // A drag must not trigger the box link underneath.
  stage.addEventListener("click", (e) => { if (moved) { e.preventDefault(); e.stopPropagation(); } }, true);

  const close = () => {
    overlay.remove();
    document.documentElement.style.overflow = "";
    document.removeEventListener("keydown", onKey);
  };
  const onKey = (e) => {
    if (e.key === "Escape") close();
    if (e.key === "+" || e.key === "=") zoomAt(1.2, ...center());
    if (e.key === "-") zoomAt(1 / 1.2, ...center());
    if (e.key === "0") fit();
  };
  document.addEventListener("keydown", onKey);
  overlay.querySelector(".kmap-tools").addEventListener("click", (e) => {
    const a = e.target.dataset.a;
    if (a === "in") zoomAt(1.25, ...center());
    if (a === "out") zoomAt(1 / 1.25, ...center());
    if (a === "fit") fit();
    if (a === "one") { scale = 1; x = 20; y = 20; apply(); }
    if (a === "close") close();
  });
  // Start readable: fit if that keeps text legible, else 75–100% from the left (step 1 side).
  const r = stage.getBoundingClientRect();
  const fitScale = Math.min(r.width / w, r.height / h) * 0.96;
  if (fitScale >= 0.75) {
    fit();
  } else {
    scale = Math.min(1, Math.max(0.75, fitScale));
    x = 24;
    y = h * scale < r.height ? (r.height - h * scale) / 2 : 24;
    apply();
  }
}

document.querySelectorAll(".kmap").forEach((el) => {
  const btn = document.createElement("button");
  btn.className = "kmap-open";
  btn.type = "button";
  btn.textContent = "⤢ Full screen";
  btn.title = "Open this map in a full-screen view";
  btn.addEventListener("click", () => openViewer(el));
  el.before(btn);
  // Double-click on the embedded map also opens it.
  el.addEventListener("dblclick", () => openViewer(el));
});
