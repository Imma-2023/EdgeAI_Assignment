import tensorflow as tf
from tensorflow.keras import layers, models

# Load dataset
dataset = tf.keras.utils.image_dataset_from_directory(
    "data/",
    image_size=(128, 128),
    batch_size=32
)
class_names = dataset.class_names
train_ds = dataset.take(70)
val_ds = dataset.skip(70)

# CNN model
model = models.Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(16, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(train_ds, validation_data=val_ds, epochs=5)
model.save("edge_ai_model.h5")
