import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping
import cv2
import numpy as np
from tensorflow.keras.metrics import mse


def load_data(image_df):
    images = []
    labels = []

    for _, row in image_df.iterrows():
        image_path = row['png_path']
        image = cv2.imread(image_path)

        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        images.append(image)

        eyeball_center = row['eyeball_center']
        center_and_radius = eval(eyeball_center)
        labels.append(center_and_radius)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    images /= 255.0

    return images, labels

def create_iris_detection_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(3))

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model


df = pd.read_csv('selected-images.csv')

images, labels = load_data(df)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=42)
X_val = X_train[int(len(X_train)*0.9):]
y_val = y_train[int(len(y_train)*0.9):]
X_train = X_train[:int(len(X_train)*0.9)]
y_train = y_train[:int(len(y_train)*0.9)]

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model = create_iris_detection_model()

history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)

print(model.summary())

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}, Test Loss: {test_loss}")

model.save('./models/vgg16_model.h5')
