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
    image = io.BytesIO(base64.b64decode(image_data))

    im = Image.open(image).convert("L")

    # Resize the image as 28x28
    im = im.resize((28, 28))

    # The pixels that is closer to 0 will be black and the pixels that is closer to 255 will be white
    im = im.point(lambda x: 0 if x<50 else 255, '1')

    return np.array(im).reshape(1, 28, 28, 1)

def predict_image(image_data):
    img = preprocess_image(image_data)
    prediction = model.predict(img)
    print(prediction)

    return TYPES[np.argmax(prediction)]