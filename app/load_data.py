import json
import numpy as np
import os
import pickle
from dotenv import load_dotenv

load_dotenv(override=True)

FEATURES_PATH = os.getenv("FEATURES_PATH", "features.pkl")
LABELS_PATH = os.getenv("LABELS_PATH", "labels.pkl")
TYPES_PATH = os.getenv("TYPES_PATH", "app/const/types.json")

files = os.listdir("data")
x_load = []
y_load = []

types = {}

def load_data():
    count = 0
    for idx, file in enumerate(files):
        print(f"Processing file {idx + 1}/{len(files)}: {file}")
        types[count] = file.split(".")[0]
        file_path = os.path.join("data", file)
        
        # Load data and handle potential memory issues by loading in smaller chunks
        x = np.load(file_path, mmap_mode='r')  # Use memory mapping for large files
        x = x.astype('float32') / 255.
        x_chunk = x[:100000, :]  # Load a chunk of data
        x_load.append(x_chunk)
        
        y_chunk = np.full((100000, 1), count, dtype='float32')
        y_load.append(y_chunk)
        
        count += 1

    # Write types to a json file
    with open(TYPES_PATH, "w") as f:
        json.dump(types, f)

    # Concatenate all the features and labels
    x_all = np.concatenate(x_load, axis=0)
    y_all = np.concatenate(y_load, axis=0)

    return x_all, y_all

# Load and process the data
features, labels = load_data()

print("Data loaded successfully. Reshaping arrays...")

# Reshape the data if needed
features = features.reshape(features.shape[0], -1).astype('float32')
labels = labels.reshape(labels.shape[0], -1).astype('float32')

print(f"Features shape: {features.shape}")
print(f"Labels shape: {labels.shape}")

# Save features and labels to pickle files
print("Saving features and labels to pickle files...")

with open(FEATURES_PATH, "wb") as f:
    pickle.dump(features, f, protocol=4)

with open(LABELS_PATH, "wb") as f:
    pickle.dump(labels, f, protocol=4)

print("Features and labels saved successfully.")