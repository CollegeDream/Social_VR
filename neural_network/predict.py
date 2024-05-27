from keras.models import load_model
import keras
import numpy as np

model = load_model('face_detection.keras')

img = keras.utils.load_img('images/distractor/Bear_00151.jpg', target_size= (180, 180))
x = keras.utils.img_to_array(img)
x = np.expand_dims(x, axis= 0)

predictions = model.predict(x)

score = float(keras.activations.sigmoid(predictions[0][0]))
print(f"This image is {100 * (1 - score):.2f}% distractor and {100 * score:.2f}% face.")