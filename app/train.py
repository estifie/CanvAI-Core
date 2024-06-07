import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pickle
from dotenv import load_dotenv
import argparse
import os
import logging
load_dotenv(override=True)

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.keras")
FEATURES_PATH = os.getenv("FEATURES_PATH", "features.pkl")
LABELS_PATH = os.getenv("LABELS_PATH", "labels.pkl")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def keras_model(image_x, image_y):
    num_of_classes = 10
    model = Sequential([
        Conv2D(32, (5, 5), input_shape=(image_x, image_y, 1), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        Conv2D(64, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.6),
        Dense(128, activation='relu'),
        Dropout(0.6),
        Dense(num_of_classes, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def load_from_pickle():
    with open(FEATURES_PATH, "rb") as f:
        features = np.array(pickle.load(f))
    with open(LABELS_PATH, "rb") as f:
        labels = np.array(pickle.load(f))
    return features, labels

def preprocess_labels(labels):
    return to_categorical(labels)

def train(epochs, batch_size):
    features, labels = load_from_pickle()
    features, labels = shuffle(features, labels)
    labels = preprocess_labels(labels)
    train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=0, test_size=0.1)
    train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
    test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)

    model = keras_model(28,28)
    model.summary()
    callbacks = [
        TensorBoard(log_dir="model")
    ]
    logger.info("Training started.")
    model.fit(train_x, train_y, batch_size=batch_size, validation_data=(test_x, test_y), epochs=epochs, callbacks=callbacks)
    model.save(MODEL_PATH)
    logger.info("Training completed. Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    
    epochs = parser.parse_args().epoch
    batch_size = parser.parse_args().batch_size

    logger.info(f"Training for {epochs} epochs with batch size of {batch_size}")

    train(epochs, batch_size)