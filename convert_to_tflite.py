import tensorflow as tf

# Load saved Keras model
model = tf.keras.models.load_model("edge_ai_model.h5")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TFLite model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
print("Model converted to TFLite successfully!")
