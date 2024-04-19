from pydantic import BaseModel

class DrawingPredict(BaseModel):
    image_data: str