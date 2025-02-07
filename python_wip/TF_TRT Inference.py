from tensorflow.python.compiler.tensorrt import trt_convert as tf_trt
from tensorflow.python.saved_model import tag_constants
import tensorflow as tf
import tensorrt as trt

import numpy as np

precision_dict = {
    "FP32": tf_trt.TrtPrecisionMode.FP32,
    "FP16": tf_trt.TrtPrecisionMode.FP16,
    "INT8": tf_trt.TrtPrecisionMode.INT8,
}

# For TF-TRT:

class OptimizedModel():
    def __init__(self, saved_model_dir = None):
        self.loaded_model_fn = None

        if not saved_model_dir is None:
            self.load_model(saved_model_dir)


    def predict(self, input_data): 
        if self.loaded_model_fn is None:
            raise(Exception("Haven't loaded a model"))
        x = tf.constant(input_data.astype('float32'))
        labeling = self.loaded_model_fn(x)
        try:
            preds = labeling['predictions'].numpy()
        except:
            try:
                preds = labeling['probs'].numpy()
            except:
                try:
                    preds = labeling[next(iter(labeling.keys()))]
                except:
                    raise(Exception("Failed to get predictions from saved model object"))
        return preds

    def load_model(self, saved_model_dir):
        saved_model_loaded = tf.saved_model.load(saved_model_dir, tags=[tag_constants.SERVING])
        wrapper_fp32 = saved_model_loaded.signatures['serving_default']

        self.loaded_model_fn = wrapper_fp32

class ModelOptimizer():
    def __init__(self, input_saved_model_dir, calibration_data=None):
        self.input_saved_model_dir = input_saved_model_dir
        self.calibration_data = None
        self.loaded_model = None

        if not calibration_data is None:
            self.set_calibration_data(calibration_data)


    def set_calibration_data(self, calibration_data):

        def calibration_input_fn():
            yield (tf.constant(calibration_data.astype('float32')), )

        self.calibration_data = calibration_input_fn


    def convert(self, output_saved_model_dir, precision="FP32", max_workspace_size_bytes=8000000000, **kwargs):

        if precision == "INT8" and self.calibration_data is None:
            raise(Exception("No calibration data set!"))

        trt_precision = precision_dict[precision]
        conversion_params = tf_trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt_precision, 
                                                                            max_workspace_size_bytes=max_workspace_size_bytes,
                                                                            use_calibration= precision == "INT8")
        converter = tf_trt.TrtGraphConverterV2(input_saved_model_dir=self.input_saved_model_dir,
                                conversion_params=conversion_params)

        if precision == "INT8":
            converter.convert(calibration_input_fn=self.calibration_data)
        else:
            converter.convert()

        converter.save(output_saved_model_dir=output_saved_model_dir)

        return OptimizedModel(output_saved_model_dir)

    def predict(self, input_data):
        if self.loaded_model is None:
            self.load_default_model()

        return self.loaded_model.predict(input_data)

    def load_default_model(self):
        self.loaded_model = tf.keras.models.load_model('resnet50_saved_model') 
        
### TF-TRT INT8 model
# Creating TF-TRT INT8 model requires a small calibration dataset. This data set ideally should represent the test data in production well, and will be used 
# to create a value histogram for each layer in the neural network for effective 8-bit quantization.

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


batch_size = 8
batched_input = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)

for i in range(batch_size):
  img_path = './data/img%d.JPG' % (i % 4)
  img = image.load_img(img_path, target_size=(224, 224))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  batched_input[i, :] = x
batched_input = tf.constant(batched_input)
print('batched_input shape: ', batched_input.shape)

def calibration_input_fn():
    yield (batched_input, )

print('Converting to TF-TRT INT8...')

converter = trt.TrtGraphConverterV2(input_saved_model_dir='resnet50_saved_model',
                                   precision_mode=trt.TrtPrecisionMode.INT8,
                                    max_workspace_size_bytes=8000000000)

converter.convert(calibration_input_fn=calibration_input_fn)
converter.save(output_saved_model_dir='resnet50_saved_model_TFTRT_INT8')
print('Done Converting to TF-TRT INT8')

def predict_tftrt(input_saved_model):
    """Runs prediction on a single image and shows the result.
    input_saved_model (string): Name of the input model stored in the current dir
    """
    img_path = './data/img0.JPG'  # Siberian_husky
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = tf.constant(x)

    saved_model_loaded = tf.saved_model.load(input_saved_model, tags=[tag_constants.SERVING])
    signature_keys = list(saved_model_loaded.signatures.keys())
    print(signature_keys)

    infer = saved_model_loaded.signatures['serving_default']
    print(infer.structured_outputs)

    labeling = infer(x)
    preds = labeling['predictions'].numpy()
    print('{} - Predicted: {}'.format(img_path, decode_predictions(preds, top=3)[0]))
    plt.subplot(2,2,1)
    plt.imshow(img);
    plt.axis('off');
    plt.title(decode_predictions(preds, top=3)[0][0][1])

def benchmark_tftrt(input_saved_model):
    saved_model_loaded = tf.saved_model.load(input_saved_model, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    N_warmup_run = 50
    N_run = 1000
    elapsed_time = []

    for i in range(N_warmup_run):
      labeling = infer(batched_input)

    for i in range(N_run):
      start_time = time.time()
      labeling = infer(batched_input)
      #prob = labeling['probs'].numpy()
      end_time = time.time()
      elapsed_time = np.append(elapsed_time, end_time - start_time)
      if i % 50 == 0:
        print('Step {}: {:4.1f}ms'.format(i, (elapsed_time[-50:].mean()) * 1000))

    print('Throughput: {:.0f} images/s'.format(N_run * batch_size / elapsed_time.sum()))

# calibration:
# from tensorflow.python.compiler.tensorrt import trt_convert as trt
# Instantiate the TF-TRT converter
# converter = trt.TrtGraphConverterV2(
#     input_saved_model_dir=SAVED_MODEL_DIR,
#     precision_mode=trt.TrtPrecisionMode.INT8,
#     use_calibration=True
# )

# # Use data from the test/validation set to perform INT8 calibration
# BATCH_SIZE=32
# NUM_CALIB_BATCHES=10
# def calibration_input_fn():
#     for i in range(NUM_CALIB_BATCHES):
#         start_idx = i * BATCH_SIZE
#         end_idx = (i + 1) * BATCH_SIZE
#         x = x_test[start_idx:end_idx, :]
#         yield [x]

# # Convert the model and build the engine with valid calibration data
# func = converter.convert(calibration_input_fn=calibration_input_fn)
# converter.build(calibration_input_fn)

# OUTPUT_SAVED_MODEL_DIR="./models/tftrt_saved_model"
# converter.save(output_saved_model_dir=OUTPUT_SAVED_MODEL_DIR)

# converter.summary()

# # Run some inferences!
# for step in range(10):
#     start_idx = step * BATCH_SIZE
#     end_idx   = (step + 1) * BATCH_SIZE

#     print(f"Step: {step}")
#     x = x_test[start_idx:end_idx, :]
#     func(x) 

# https://github.com/tensorflow/tensorrt/blob/master/tftrt/examples-py/PTQ_example.ipynb
model = tf.keras.models.load_model('resnet50_saved_model')

batch_size = 8
batched_input = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)

for i in range(batch_size):
  img_path = './data/img%d.JPG' % (i % 4)
  img = image.load_img(img_path, target_size=(224, 224))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  batched_input[i, :] = x
batched_input = tf.constant(batched_input)
print('batched_input shape: ', batched_input.shape)

# Benchmarking throughput
N_warmup_run = 50
N_run = 1000
elapsed_time = []

for i in range(N_warmup_run):
  preds = model.predict(batched_input)

for i in range(N_run):
  start_time = time.time()
  preds = model.predict(batched_input)
  end_time = time.time()
  elapsed_time = np.append(elapsed_time, end_time - start_time)
  if i % 50 == 0:
    print('Step {}: {:4.1f}ms'.format(i, (elapsed_time[-50:].mean()) * 1000))

print('Throughput: {:.0f} images/s'.format(N_run * batch_size / elapsed_time.sum()))

def predict_tftrt(input_saved_model):
    """Runs prediction on a single image and shows the result.
    input_saved_model (string): Name of the input model stored in the current dir
    """
    img_path = './data/img0.JPG'  # Siberian_husky
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = tf.constant(x)

    saved_model_loaded = tf.saved_model.load(input_saved_model, tags=[tag_constants.SERVING])
    signature_keys = list(saved_model_loaded.signatures.keys())
    print(signature_keys)

    infer = saved_model_loaded.signatures['serving_default']
    print(infer.structured_outputs)

    labeling = infer(x)
    preds = labeling['predictions'].numpy()
    print('{} - Predicted: {}'.format(img_path, decode_predictions(preds, top=3)[0]))
    plt.subplot(2,2,1)
    plt.imshow(img);
    plt.axis('off');
    plt.title(decode_predictions(preds, top=3)[0][0][1])

def benchmark_tftrt(input_saved_model):
    saved_model_loaded = tf.saved_model.load(input_saved_model, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    N_warmup_run = 50
    N_run = 1000
    elapsed_time = []

    for i in range(N_warmup_run):
      labeling = infer(batched_input)

    for i in range(N_run):
      start_time = time.time()
      labeling = infer(batched_input)
      end_time = time.time()
      elapsed_time = np.append(elapsed_time, end_time - start_time)
      if i % 50 == 0:
        print('Step {}: {:4.1f}ms'.format(i, (elapsed_time[-50:].mean()) * 1000))

    print('Throughput: {:.0f} images/s'.format(N_run * batch_size / elapsed_time.sum()))