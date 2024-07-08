import numpy as np
import tensorflow as tf
import keras
from keras import Sequential, layers
import matplotlib.pyplot as plt
import cv2

# Generators
train_ds = keras.utils.image_dataset_from_directory(
    directory='train',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256)
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory='test',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256)
)


# Normalize
def process(image, label):
    image = tf.cast(image/255., tf.float32)
    return image, label


train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)

# Create CNN Model
model = Sequential()

model.add(layers.Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(256, 256, 3)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

model.add(layers.Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

model.add(layers.Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

model.add(layers.Conv2D(256, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

model.add(layers.Flatten())

model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, epochs=10, validation_data=validation_ds)

# Visualizing the performance of the CNN Model
plt.plot(history.history['accuracy'], color='red', label='train')
plt.plot(history.history['val_accuracy'], color='blue', label='validation')
plt.legend()
plt.show()

plt.plot(history.history['loss'], color='red', label='train')
plt.plot(history.history['val_loss'], color='blue', label='validation')
plt.legend()
plt.show()

# Testing input images
test_img = cv2.imread('dog.jpg')
test_img = cv2.resize(test_img, (256, 256))
test_input = test_img.reshape((1, 256, 256, 3))
model.predict(test_input)
plt.imshow(test_img)
plt.show()

test_img = cv2.imread('cat.jpg')
test_img = cv2.resize(test_img, (256, 256))
test_input = test_img.reshape((1, 256, 256, 3))
model.predict(test_input)
plt.imshow(test_img)
plt.show()


# Testing function to preprocess and predict image class
def predict_and_plot(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (256, 256))
    img_normalized = img_resized / 255.0
    test_input = np.expand_dims(img_normalized, axis=0)
    predictions = model.predict(test_input)
    predicted_class = 'Dog' if predictions[0] >= 0.5 else 'Cat'
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f'Prediction: {predicted_class}')
    plt.axis('off')
    plt.show()


# Test the model with images
predict_and_plot('dog.jpg')
predict_and_plot('cat.jpg')
