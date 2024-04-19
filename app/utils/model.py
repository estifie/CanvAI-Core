from keras.models import load_model as load_model_keras

def load_model(model_path):
    model = load_model_keras(model_path)
    return model