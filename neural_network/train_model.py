##################################################################################
# taken from: https://keras.io/examples/vision/image_classification_from_scratch/
#
# To optimize structure of NN:
# https://github.com/keras-team/keras-tuner
#
# face images:
# https://vis-www.cs.umass.edu/fddb/
#
# distractor images:
# https://ufdd.info/
#
# !!!!!!! TRAINING A NEURAL NETWORK TAKES A LONG TIME
# !!!!!!! ONLY RUN THIS IF YOU HAVE CHANGED THE LAYERS/STRUCTURE OF THE NETWORK
##################################################################################

import os
# keras library for creating neural networks
# dependencies: numpy, pandas, scikit-learn, matplotlib, scipy, seaborn
# pip install keras
# python -m pip show keras
import keras
from keras import layers
# tensorflow library for data management
# pip install tensorflow
from tensorflow import data as tf_data

# clean data, get rid of any corrupted images
num_skipped = 0
for folder_name in ("distractor", "face"):
    folder_path = os.path.join("images", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = b"JFIF" in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

# set image, batch size
image_size = (180, 180)
batch_size = 128

# load and split data from UFDD
train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "images",
    validation_split=0.2,
    subset="both",
    seed=57,
    image_size=image_size,
    batch_size=batch_size,
)

# rotate/flip data to combat overfitting
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

# Apply `data_augmentation` to the training images.
train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf_data.AUTOTUNE,
)

# Prefetching samples in GPU memory helps maximize GPU utilization.
train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    ### Commenting out the residuals because it seems the neural network works better without them
    # previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        ### Commenting out the residuals because it seems the neural network works better without them
        # # Project residual
        # residual = layers.Conv2D(size, 1, strides=2, padding="same")(
        #     previous_block_activation
        # )
        # x = layers.add([x, residual])  # Add back residual
        # previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    x = layers.Dropout(0.25)(x)
    # We specify activation=None so as to return logits
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)

# increase num_classes if you want to classify more images than just face/not face
model = make_model(input_shape=image_size + (3,), num_classes=2)
# we could save a plot of the model in the future??
# keras.utils.plot_model(model, show_shapes=True)

# compile and train the model
# should never go higher than ~50
epochs = 1

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]
model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)
model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)

# model can be loaded in other code by:
## from keras.models import load_model
## model = load_model('face_detection.keras')
# once loaded, to predict an image class using the model:
## from keras.preprocessing import image
## import numpy as np
## img = image.load_img('img_name.jpg', target_size= (width, height))
## x = image.img_to_array(img)
## x = np.expand_dims(x, axis= 0)
## images = np.vstack([x])
## classes = model.predict_classes(images, batch_size= 128)
model.save('face_detection.keras')