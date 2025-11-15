const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let drawing = false;


// ------------------ INITIAL BLACK BACKGROUND ------------------
function resetCanvas() {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}
resetCanvas();


function resizeCanvasForHiDPI() {
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;

  // Resizing clears the canvas — unavoidable
  canvas.width = rect.width * scale;
  canvas.height = rect.height * scale;

  // Reset scale
  ctx.scale(scale, scale);

  // Redraw the background
  resetCanvas()
}

resizeCanvasForHiDPI();
window.addEventListener("resize", resizeCanvasForHiDPI);


// ------------------ POSITION HELPERS ------------------
function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;

  if (e.touches) {
    return {
      x: (e.touches[0].clientX - rect.left) * scaleX,
      y: (e.touches[0].clientY - rect.top) * scaleY
    };
  }

  return {
    x: e.offsetX * scaleX,
    y: e.offsetY * scaleY
  };
}

// ------------------ DRAWING EVENTS ------------------
function startDraw(e) {
  e.preventDefault();
  drawing = true;

  const { x, y } = getPos(e);
  ctx.beginPath();
  ctx.moveTo(x, y);
}

function moveDraw(e) {
  if (!drawing) return;

  const { x, y } = getPos(e);

  ctx.lineWidth = 40;
  ctx.lineCap = "round";
  ctx.strokeStyle = "white";

  ctx.lineTo(x, y);
  ctx.stroke();
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
  // Downscale to 280x280 for backend
  const small = document.createElement('canvas');
  small.width = 280;
  small.height = 280;
  const sctx = small.getContext('2d');

  sctx.drawImage(canvas, 0, 0, 280, 280);

  const dataUrl = small.toDataURL('image/png');

  const response = await fetch(
    'https://personal-website-40217740204.europe-west1.run.app/digit-predict',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataUrl })
    }
  );

  const result = await response.json();
  
  console.log(result)
  const probs = result.probabilities;

  const predDigit = data.probabilities.indexOf(Math.max(...data.probabilities));
  document.getElementById('result').innerText = `Prediction: ${predDigit}`;

  drawChart(probs);
});


// ------------------ Draw Function ------------------

function drawChart(probabilities) {
  const chart = document.getElementById("chart");
  chart.innerHTML = "";  // Clear previous

  const maxProb = Math.max(...probabilities);

  probabilities.forEach((p, digit) => {
    const bar = document.createElement("div");
    bar.className = "bar";
    bar.style.height = `${(p / maxProb) * 100}%`;

    const label = document.createElement("div");
    label.className = "bar-label";
    label.innerText = digit;

    const value = document.createElement("div");
    value.className = "bar-value";
    value.innerText = p.toFixed(2);

    bar.appendChild(value);
    bar.appendChild(label);
    chart.appendChild(bar);
  });
}