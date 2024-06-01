from fastapi import APIRouter, HTTPException
from dotenv import load_dotenv
from app.utils.predict import predict_image
from app.schemas.drawing import DrawingPredict
import os
load_dotenv(override=True)

prefix = "full_numpy_bitmap_"

router = APIRouter(
    prefix="/model",
    tags=["model"],
    responses={404: {"status": "error", "message": "Not found"}},
)

@router.post("/predict")
async def predict(data: DrawingPredict):
    try:
        prediction, probability = predict_image(data.image_data)

        if prediction.startswith(prefix):
            prediction = prediction[len(prefix):]

        return {"status": "success", "data": {
            "prediction": prediction,
            "probability": probability
        }}
    except Exception as e:
        raise HTTPException(status_code=400, detail={"status": "error", "message": str(e)})
    