import base64
import io
import numpy as np
from PIL import Image
import onnxruntime as ort

# Load ONNX model once
session = ort.InferenceSession("smallcnn.onnx")
input_name = session.get_inputs()[0].name

def predict_digit(image_base64):
    img_data = image_base64.split(",")[1]

    img = (
        Image.open(io.BytesIO(base64.b64decode(img_data)))
        .convert("L")
        .resize((28, 28))
    )

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.reshape(1, 1, 28, 28)

    # Run ONNX inference
    outputs = session.run(None, {input_name: arr})
    probs = outputs[0]

    pred = int(np.argmax(probs))
    return {"digit": pred}
