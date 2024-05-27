from keras.models import load_model
import keras
import numpy as np

model = load_model('face_detection.keras')

model.summary()