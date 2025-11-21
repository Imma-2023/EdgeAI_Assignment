import numpy as np
import tensorflow as tf
from PIL import Image

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess image
img = Image.open("sample.jpg").resize((128,128))
input_data = np.expand_dims(np.array(img)/255.0, axis=0).astype(np.float32)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get prediction
output = interpreter.get_tensor(output_details[0]['index'])
prediction = np.argmax(output)
print("Predicted class index:", prediction)
