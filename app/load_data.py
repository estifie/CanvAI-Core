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
x = []
x_load = []
y = []
y_load = []

types = {}

def load_data():
    count = 0
    for file in files:
        types[count] = file.split(".")[0]
        file = "data/" + file
        x = np.load(file)
        x = x.astype('float32') / 255.
        x = x[0:100000, :]
        x_load.append(x)
        y = [count for _ in range(100000)]
        count += 1
        y = np.array(y).astype('float32')
        y = y.reshape(y.shape[0], 1)
        y_load.append(y)

    # Write types to a json file
    with open("app/const/types.json", "w") as f:
        json.dump(types, f)

    return x_load, y_load


features, labels = load_data()
features = np.array(features).astype('float32')
labels = np.array(labels).astype('float32')
features=features.reshape(features.shape[0]*features.shape[1],features.shape[2])
labels=labels.reshape(labels.shape[0]*labels.shape[1],labels.shape[2])

with open(FEATURES_PATH, "wb") as f:
    pickle.dump(features, f, protocol=4)
with open(LABELS_PATH, "wb") as f:
    pickle.dump(labels, f, protocol=4)