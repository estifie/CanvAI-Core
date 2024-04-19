from fastapi import APIRouter, HTTPException
from dotenv import load_dotenv
from app.utils.predict import predict_image
from app.schemas.drawing import DrawingPredict
import os
load_dotenv(override=True)

router = APIRouter(
    prefix="/model",
    tags=["model"],
    responses={404: {"status": "error", "message": "Not found"}},
)

@router.post("/predict")
async def predict(data: DrawingPredict):
    try:

        prediction = predict_image(data.image_data)
        print(prediction)
        return {"status": "success", "data": {
            "prediction": prediction
        }}
    except Exception as e:
        raise HTTPException(status_code=400, detail={"status": "error", "message": str(e)})
    