from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import base64
import io
from PIL import Image
import numpy as np
import torch  # or tensorflow

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    img_data = data["image"].split(",")[1]  # remove data URL prefix
    img = Image.open(io.BytesIO(base64.b64decode(img_data))).convert("L").resize((28, 28))
    arr = np.array(img) / 255.0
    # prediction = model.predict(arr.reshape(1, 1, 28, 28))
    prediction = np.random.randint(0, 10)  # placeholder
    return {"digit": int(prediction)}