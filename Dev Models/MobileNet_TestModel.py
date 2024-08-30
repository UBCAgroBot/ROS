import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

"""
Mobilenet has a couple of versions. v2 is supposed to have a good balance between efficiency and performance, and simpler than using v3.
v3 is supposed to be an improvement, and has 2 variants. The large version is supposed to have high accuracy, and the small variant is supposed to be optimized for
resource-constrained environments.
v1 is good for a resource-constrained environment, but v2 and v3 should be improved versions of the original.
Model will be based on mobilenetv2
"""

# Load the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape input data to have a single channel (grayscale)
x_train_resized = x_train.reshape((-1, 28, 28, 1))
x_test_resized = x_test.reshape((-1, 28, 28, 1))

# Convert grayscale images to 3 channels (RGB) and resize to 64x64
def preprocess_images(images):
    images_rgb = tf.image.grayscale_to_rgb(tf.convert_to_tensor(images, dtype=tf.float32))
    images_rgb_resized = tf.image.resize(images_rgb, (64, 64))
    return images_rgb_resized

x_train_rgb_resized = preprocess_images(x_train_resized)
x_test_rgb_resized = preprocess_images(x_test_resized)

# Define the MobileNet model with additional layers
def create_mobilenet_model(input_shape=(64, 64, 3), num_classes=10):
    model = models.Sequential()

    # MobileNetV2 base
    mobilenetv2_base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
    )
    # Freeze the MobileNetV2 base
    mobilenetv2_base.trainable = False
    model.add(mobilenetv2_base)

    # Add Batch Normalization after MobileNetV2
    model.add(layers.BatchNormalization())

    # GlobalAveragePooling layer instead of flatten layer
    model.add(layers.GlobalAveragePooling2D())

    # Add Dropout before Dense layers
    model.add(layers.Dropout(0.5))

    # Add Dense layers with L2 regularization
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))

    # Responsible for output, 10 neurons so we have 1 neuron per class (t-shirt, trouser, etc)
    model.add(layers.Dense(10, activation='softmax'))

    return model

# Create the MobileNet model
mobilenetv2_model = create_mobilenet_model()

# Compile the model
mobilenetv2_model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

# Define early stopping callback, also restore best performing weights
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, min_delta=0.001)

# Train the model without data augmentation
mobilenetv2_model.fit(x_train_rgb_resized, y_train, epochs=20, validation_data=(x_test_rgb_resized, y_test), callbacks=[early_stopping])

# Save the trained model
mobilenetv2_model.save('mobilenetv2_model.h5')

# Evaluate the model on the test set
test_loss, test_acc = mobilenetv2_model.evaluate(x_test_rgb_resized, y_test)
print(f'Test accuracy: {test_acc}')

import tensorflow as tf
import numpy as np

# Load the saved model
loaded_model = tf.keras.models.load_model('mobilenetv2_model2.h5')

# Load the Fashion MNIST test data
(_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Preprocess the test data
x_test = x_test / 255.0
x_test = np.expand_dims(x_test, axis=-1)

# Resize the images to match the input shape of the model
x_test_resized = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_test, dtype=tf.float32))
x_test_resized = tf.image.resize(x_test_resized, (64, 64))

# Evaluate the model on the test set
test_loss, test_acc = loaded_model.evaluate(x_test_resized, y_test)
print(f'Test accuracy: {test_acc}')

# Choose the number of random samples to test
num_samples = 5

# Randomly select samples and make predictions
for _ in range(num_samples):
    sample_index = np.random.randint(0, len(x_test_resized))
    sample_image = x_test_resized[sample_index]
    sample_label = y_test[sample_index]

    # Reshape the image to match the model input shape
    sample_image = np.expand_dims(sample_image, axis=0)

    # Make predictions
    predictions = loaded_model.predict(sample_image)

    # Get the predicted class
    predicted_class = np.argmax(predictions)
    print(f'Actual Label: {sample_label}, Predicted Label: {predicted_class}')

from google.colab import files

# Save the model file
model_filename = 'mobilenetv2_model2.h5'
loaded_model.save(model_filename)

# Download the saved model file
files.download(model_filename)
