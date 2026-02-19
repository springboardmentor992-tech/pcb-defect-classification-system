import os
import torch
import torch.nn as nn
from flask import Flask, render_template, request
from torchvision import transforms
from PIL import Image

# -------------------- Flask Setup --------------------
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# -------------------- Device --------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# -------------------- CNN Model --------------------
class PCB_CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# -------------------- Load Model --------------------
checkpoint = torch.load("pcb_cnn_path.pth", map_location=device)
class_names = checkpoint["class_names"]

model = PCB_CNN(len(class_names)).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# -------------------- Transform --------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# -------------------- Prediction Function --------------------
def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return class_names[pred], probs[0][pred].item()

# -------------------- Routes --------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    template_path = None
    test_path = None

    if request.method == "POST":
        template_img = request.files["template"]
        test_img = request.files["test"]

        template_path = os.path.join(app.config["UPLOAD_FOLDER"], template_img.filename)
        test_path = os.path.join(app.config["UPLOAD_FOLDER"], test_img.filename)

        template_img.save(template_path)
        test_img.save(test_path)

        prediction, confidence = predict_image(test_path)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        template_path=template_path,
        test_path=test_path
    )

if __name__ == "__main__":
    app.run(debug=True)
