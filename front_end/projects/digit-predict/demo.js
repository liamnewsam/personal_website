// demo.js — example small interactive component

document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("demoBtn");
  const out = document.getElementById("demoOutput");

  if (!btn || !out) return;

  let count = 0;

  btn.addEventListener("click", () => {
    count++;
    out.textContent = `You clicked ${count} time${count === 1 ? "" : "s"}!`;
  });
});
