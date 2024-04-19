from fastapi import FastAPI
from dotenv import load_dotenv
from app.routers import predict
from app.utils.model import load_model
import os
load_dotenv(override=True)

BUILD = os.getenv("BUILD", "dev")
VERSION = os.getenv("VERSION", "v1")
PORT = os.getenv("PORT", 8000)
root_prefix = f"/api/{VERSION}"

app = FastAPI(title="CanvAI Core API", version=VERSION)

app.include_router(predict.router, prefix=root_prefix)
