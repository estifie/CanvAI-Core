import json
import numpy as np
from PIL import Image
from app.utils.model import load_model
import base64
import io
from dotenv import load_dotenv
import os

load_dotenv(override=True)

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.keras")
model = load_model(MODEL_PATH)

# Load types.json to TYPES
TYPES_PATH = os.getenv("TYPES_PATH", "app/const/types.json")
with open(TYPES_PATH, "r") as f:
    TYPES = json.load(f)


def preprocess_image(image_data):
    decoded_data = base64.b64decode(image_data)
    
    image = Image.open(io.BytesIO(decoded_data)).convert("L")

    img_array = np.array(image)

    return img_array.reshape(1, 28, 28, 1)

def predict_image(image_data):
    img = preprocess_image(image_data)

    prediction = model.predict(img)
    
    return TYPES[str(np.argmax(prediction))], float(np.max(prediction))