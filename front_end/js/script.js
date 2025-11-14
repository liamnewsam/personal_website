/* ============================================================
   script.js — project renderer (path-based)
   - Edit the `PROJECTS` array below.
   - Each project: id, title, desc, thumb, color (one of the theme accents)
   - Tiles link to: /projects/<id>/
   ============================================================ */

const PROJECTS = [
  {
    id: "digit-predict",
    title: "Digit Classifier",
    desc: "Small CNN with visual explanations and demo.",
    thumb: "images/digit.png",
    color: "var(--accent-1)"
  },
  {
    id: "housing-study",
    title: "Housing Study",
    desc: "Participant dashboard and automated reports.",
    thumb: "images/housing.png",
    color: "var(--accent-4)"
  }
];

/* create element helper */
function el(tag, opt = {}) {
  const e = document.createElement(tag);
  if (opt.cls) e.className = opt.cls;
  if (opt.text) e.textContent = opt.text;
  if (opt.html) e.innerHTML = opt.html;
  if (opt.attrs) Object.entries(opt.attrs).forEach(([k,v])=>e.setAttribute(k,v));
  return e;
}

/* render the grid */
function renderProjects() {
  const grid = document.getElementById("projectGrid");
  if (!grid) return;

  grid.innerHTML = ""; // reset

  PROJECTS.forEach(p => {
    const a = el("a", { cls: "project-tile", attrs: { href: `/projects/${encodeURIComponent(p.id)}/`, role: "listitem" }});

    // title stripe color: apply to pseudo-element via inline style variable
    a.style.setProperty("--tile-accent", p.color || "var(--accent-1)");
    // set a data attribute so CSS can read it if needed
    a.setAttribute("data-accent", p.color || "var(--accent-1)");

    // injection of a small colored bar (we use ::before in CSS but also set style here)
    a.querySelectorAll = a.querySelectorAll; // noop so some linters don't complain

    const stripe = el("div", { cls: "tile-stripe" });
    stripe.style.height = "6px";
    stripe.style.borderRadius = "4px";
    stripe.style.background = p.color || "var(--accent-1)";
    stripe.style.marginBottom = "6px";

    const thumb = el("img", { cls: "project-thumb", attrs: { src: p.thumb, alt: `${p.title} screenshot` }});
    const title = el("h4", { cls: "project-title", text: p.title });
    const desc = el("p", { cls: "project-desc", text: p.desc });

    a.appendChild(stripe);
    a.appendChild(thumb);
    a.appendChild(title);
    a.appendChild(desc);

    grid.appendChild(a);
  });
}

/* init */
document.addEventListener("DOMContentLoaded", () => {
  renderProjects();
});
