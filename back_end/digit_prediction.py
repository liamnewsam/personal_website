
import base64
import io
from PIL import Image
import numpy as np
import torch  # or tensorflow
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(2),                 
            nn.Conv2d(8, 64, 3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2),               
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 256),  # Adjusted based on the output size of the features block
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# 1. Create the model architecture
model_loaded = SmallCNN()

# 2. Load the saved parameters
model_loaded.load_state_dict(torch.load("smallcnn.pth"))

model_loaded.eval()


def predict_digit(image_base64):
    img_data = image_base64.split(",")[1]  # remove data URL prefix
    img = Image.open(io.BytesIO(base64.b64decode(img_data))).convert("L").resize((28, 28))
    arr = np.array(img, dtype=np.float32) / 255.0
    logits = model_loaded(arr.reshape(1, 1, 28, 28))
    probs = F.softmax(logits, dim=1)
    print(probs)
    return {"digit": probs.argmax(dim=1).item()}
