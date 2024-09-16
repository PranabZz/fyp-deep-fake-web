# deepfake_detector/model_loader.py

import tensorflow as tf

def load_model():
    model_path = '/home/pranab/Desktop/Documents/fyp/myproject/cnn_rnn_deepfake_model.h5'
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()
