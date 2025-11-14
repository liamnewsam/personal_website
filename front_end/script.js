/* ============================================================
   PROJECT CONFIGURATION (Edit as needed)
   ------------------------------------------------------------
   - Add/modify projects by updating the `projects` array.
   - Each project needs: id, title, short, thumb.
   - Clicking a tile navigates to /projects/<id>/.
   ============================================================ */

const projects = [
  {
    id: "sys-tools",
    title: "OS Tools",
    short: "CLI tools for process scheduling experiments.",
    thumb: "images/sys-tools.png"
  },
  {
    id: "digit-predict",
    title: "Digit Classifier",
    short: "Small CNN with explainable visualizations.",
    thumb: "images/digit.png"
  },
  {
    id: "drawing-canvas",
    title: "Canvas Draw",
    short: "A responsive drawing canvas with export.",
    thumb: "images/canvas.png"
  },
  {
    id: "housing-study",
    title: "Housing Study",
    short: "Participant dashboard and automated reports.",
    thumb: "images/housing.png"
  }
];

/* ============================================================
   ELEMENT CREATION UTIL
   ============================================================ */
function el(tag, opts = {}) {
  const e = document.createElement(tag);
  if (opts.class) e.className = opts.class;
  if (opts.text) e.textContent = opts.text;
  if (opts.html) e.innerHTML = opts.html;
  if (opts.attrs) {
    for (const [k, v] of Object.entries(opts.attrs)) {
      e.setAttribute(k, v);
    }
  }
  return e;
}

/* ============================================================
   RENDER PROJECT GRID
   ============================================================ */
function renderProjects() {
  const grid = document.getElementById("projectGrid");
  grid.innerHTML = "";

  projects.forEach(p => {
    const tile = el("a", {
      class: "project-tile",
      attrs: { href: `/projects/${p.id}/` }
    });

    const thumb = el("img", {
      class: "project-thumb",
      attrs: { src: p.thumb, alt: `${p.title} screenshot` }
    });

    const title = el("h3", {
      class: "project-title",
      text: p.title
    });

    const meta = el("div", {
      class: "project-meta",
      text: p.short
    });

    tile.appendChild(thumb);
    tile.appendChild(title);
    tile.appendChild(meta);

    grid.appendChild(tile);
  });
}

/* ============================================================
   INIT
   ============================================================ */
document.addEventListener("DOMContentLoaded", () => {
  renderProjects();
});
