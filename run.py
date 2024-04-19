import argparse
import subprocess
from dotenv import load_dotenv
import os
import sys
load_dotenv(override=True)

PORT = os.getenv("PORT", "8000")

def app():
    cmd = [
        "uvicorn",
        "app.main:app",
        "--reload",
        "--port",
        PORT
        
    ]
    subprocess.run(cmd)

def load():
    cmd = [
        sys.executable,
        "app/load_data.py"
    ]
    subprocess.run(cmd)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "app/train.py",
        "--epoch",
        str(args.epoch),
        "--batch_size",
        str(args.batch_size)
    ]
    subprocess.run(cmd)

    
