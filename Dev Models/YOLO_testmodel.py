import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 1: Load and Preprocess Fashion MNIST Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,  # random rotation up to 10 degrees
    width_shift_range=0.1,  # randomly shift width by up to 10%
    height_shift_range=0.1,  # randomly shift height by up to 10%
    horizontal_flip=True,  # randomly flip horizontally
    shear_range=0.2,  # shear transformation
    zoom_range=0.2,  # random zoom
    fill_mode="nearest"  # fill mode
)

# Apply data augmentation to x_train
x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
train_generator = datagen.flow(x_train, tf.keras.utils.to_categorical(y_train, 10), batch_size=64)

# Preprocess x_test
x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Convert labels to categorical
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Step 2: Design YOLO Model for Classification
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Step 3: Compile the Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 4: Train the Model using the generator
model.fit(train_generator, epochs=20, validation_data=(x_test, y_test))

# Step 5: Save the trained model
model.save("fashion_mnist_model.h5")

# Later, if you want to load and evaluate the model without retraining
loaded_model = tf.keras.models.load_model("fashion_mnist_model.h5")

# Evaluate the loaded model on the test set
loaded_model.evaluate(x_test, y_test)

# Step 6: Evaluate the Model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')

# Later, if you want to load and evaluate the model without retraining
loaded_model = tf.keras.models.load_model("fashion_mnist_model.h5")

# Evaluate the loaded model on the test set
loaded_model.evaluate(x_test, y_test)

# Step 6: Evaluate the Model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')
