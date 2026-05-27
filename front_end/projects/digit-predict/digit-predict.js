const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let drawing = false;
let lastX = 0;
let lastY = 0;


// ------------------ INITIAL BLACK BACKGROUND ------------------
function resetCanvas() {
  // Use rect dimensions (CSS px) since ctx is scaled to CSS-pixel space
  const rect = canvas.getBoundingClientRect();
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, rect.width, rect.height);
}


function resizeCanvasForHiDPI() {
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;

  // Setting canvas.width resets context state (including transforms) — unavoidable
  canvas.width = rect.width * scale;
  canvas.height = rect.height * scale;

  // Scale so all drawing calls use CSS-pixel coordinates
  ctx.scale(scale, scale);

  resetCanvas();
}

resizeCanvasForHiDPI();
window.addEventListener("resize", resizeCanvasForHiDPI);


// ------------------ POSITION HELPERS ------------------
// Always return CSS-pixel coordinates to match the scaled ctx
function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  const clientX = e.touches ? e.touches[0].clientX : e.clientX;
  const clientY = e.touches ? e.touches[0].clientY : e.clientY;
  return {
    x: clientX - rect.left,
    y: clientY - rect.top
  };
}

// ------------------ DRAWING EVENTS ------------------
function startDraw(e) {
  e.preventDefault();
  drawing = true;

  const { x, y } = getPos(e);
  lastX = x;
  lastY = y;

  // Draw a dot so single clicks/taps register
  ctx.beginPath();
  ctx.arc(x, y, 10, 0, Math.PI * 2);
  ctx.fillStyle = "white";
  ctx.fill();
}

function moveDraw(e) {
  if (!drawing) return;
  e.preventDefault();

  const { x, y } = getPos(e);

  // Begin a new path per segment to avoid re-stroking the entire history
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(x, y);
  ctx.lineWidth = 20;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "white";
  ctx.stroke();

  lastX = x;
  lastY = y;
}

function endDraw() {
  drawing = false;
}

canvas.addEventListener('mousedown', startDraw);
canvas.addEventListener('mousemove', moveDraw);
canvas.addEventListener('mouseup', endDraw);
canvas.addEventListener('mouseleave', endDraw);

canvas.addEventListener('touchstart', startDraw, { passive: false });
canvas.addEventListener('touchmove', moveDraw, { passive: false });
canvas.addEventListener('touchend', endDraw);
canvas.addEventListener('touchcancel', endDraw);

// ------------------ ERASE BUTTON ------------------
document.getElementById('erase').addEventListener('click', resetCanvas);

// ------------------ PREDICT BUTTON ------------------
document.getElementById('predict').addEventListener('click', async () => {
  const btn = document.getElementById('predict');
  btn.disabled = true;
  btn.textContent = "Predicting…";

  // Downscale to 280x280 for backend
  const small = document.createElement('canvas');
  small.width = 280;
  small.height = 280;
  const sctx = small.getContext('2d');

  sctx.drawImage(canvas, 0, 0, 280, 280);

  const dataUrl = small.toDataURL('image/png');

  try {
    const response = await fetch(
      'https://personal-website-40217740204.europe-west1.run.app/digit-predict',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl })
      }
    );

    const result = await response.json();

    const probs = result.probabilities;
    const predDigit = probs.indexOf(Math.max(...probs));
    document.getElementById('result').innerText = `Prediction: ${predDigit}`;

    updateChart(probs);
  } catch (err) {
    document.getElementById('result').innerText = 'Error — try again';
  } finally {
    btn.disabled = false;
    btn.textContent = "Predict";
  }
});


// ------------------ CHART ------------------

function initializeChart() {
  const chart = document.getElementById("prob-chart");
  chart.innerHTML = "";

  for (let i = 0; i < 10; i++) {
    const row = document.createElement("div");
    row.className = "bar-row";

    const label = document.createElement("span");
    label.className = "bar-label";
    label.textContent = i;

    const track = document.createElement("div");
    track.className = "bar-track";

    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.dataset.index = i;
    track.appendChild(fill);

    const pct = document.createElement("span");
    pct.className = "bar-pct";
    pct.dataset.pct = i;
    pct.textContent = "0.000";

    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(pct);
    chart.appendChild(row);
  }
}

initializeChart();


function updateChart(probabilities) {
  const maxIndex = probabilities.indexOf(Math.max(...probabilities));

  probabilities.forEach((p, i) => {
    const fill = document.querySelector(`.bar-fill[data-index="${i}"]`);
    fill.style.width = `${(p * 100).toFixed(1)}%`;
    fill.classList.toggle("highlight", i === maxIndex);

    const pct = document.querySelector(`.bar-pct[data-pct="${i}"]`);
    pct.textContent = p.toFixed(3);
  });
}
