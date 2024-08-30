!pip install onnx
!pip install keras2onnx

import onnx
import keras2onnx
import tensorflow as tf
import numpy as np

from keras import layers, models
from keras.applications import ResNet50
from keras.datasets import fashion_mnist
from keras.utils import to_categorical

from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load dataset/Preprocessing
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Record batch size
batch_size_var = len(train_images) // 64

# Normalize image values
train_images = train_images / 255.0
test_images = test_images / 255.0

# Convert images to RGB from grayscale (add channel dimension)
train_images = tf.image.grayscale_to_rgb(tf.expand_dims(train_images, axis=-1))
test_images = tf.image.grayscale_to_rgb(tf.expand_dims(test_images, axis=-1))

# Resize images to 32x32
train_images = tf.image.resize(train_images, (32, 32))
test_images = tf.image.resize(test_images, (32, 32))

# Convert labels to one-hot encoding
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

# Instantiate model architecture
model = models.Sequential()

# Load the pre-trained EfficientNetB0 model backbone
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(32, 32, 3),
    pooling='avg',
    classes=10,
)

# Freeze the convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# Add output layers to model
model.add(base_model)
model.add(layers.Flatten())
# model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(1024, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation="softmax"))

# cnn.add(BatchNormalization())
# cnn.add(Dropout(0.2))
# cnn.add(Conv2D(32, kernel_size=3, activation="relu", padding="same"))
# cnn.add(Dropout(0.2))
# cnn.add(Conv2D(24, kernel_size=3, activation="relu", padding="same"))
# cnn.add(Dropout(0.2))
# cnn.add(Conv2D(64, kernel_size=3, activation="relu", padding="same"))
# cnn.add(MaxPooling2D(pool_size=(2, 2)))
# cnn.add(Dropout(0.2))
# cnn.add(Flatten())
# cnn.add(Dense(128, activation="relu"))
# cnn.add(Dropout(0.3))
# cnn.add(Dense(10, activation="softmax"))

# Display the model summary
model.summary()

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Training model
model.fit(
    train_images,
    train_labels,
    steps_per_epoch=batch_size_var,
    epochs=10,
    batch_size=64,
    validation_data=(test_images, test_labels),
    verbose=1,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
    ],
)

# Evaluate performance
test_loss, test_acc = model.evaluate(test_images, test_labels)

# Make predictions on test dataset
predictions = model.predict(test_images)

# Normalize (one-hot encode values)
predictions = np.argmax(predictions, axis=1)
test_labels = np.argmax(test_labels, axis=1)

# Printing results
print(f"Accuracy: {test_acc:.4f}")
print(f"Loss: {test_loss:.4f}")
print(f"Precision: {precision_score(test_labels, predictions, average='weighted'):.4f}")
print(f"Recall: {recall_score(test_labels, predictions, average='weighted'):.4f}")

# Saving model
model.save("FRCNN.h5")
FRCNN_file = keras2onnx.convert_keras(model, model.name, target_opset=12)
onnx.save_model(FRCNN_file, "FRCNN.onxx")

# k-fold analysis, 78% -> 90%
# option to load model in one block, then run analytics in another
# change Learning rate?, conv / pooling layers
# last test: 60% acc, maybe unfreeze layers?
