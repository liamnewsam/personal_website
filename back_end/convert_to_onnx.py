import torch
from digit_prediction import SmallCNN

model = SmallCNN()
model.load_state_dict(torch.load("smallcnn.pth", map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 1, 28, 28)
torch.onnx.export(
    model,
    dummy_input,
    "smallcnn.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=11
)

print("Converted to ONNX!")