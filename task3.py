# ===============================
# Handwritten Character Recognition
# Single Combined Code
# ===============================

# Import libraries
import numpy as np
import cv2

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import classification_report


# ===============================
# Step 1: Load and Preprocess Data
# ===============================

def load_and_preprocess():

    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test  = x_test.astype('float32') / 255.0

    # Reshape for CNN
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test  = x_test.reshape(-1, 28, 28, 1)

    return x_train, y_train, x_test, y_test


# ===============================
# Step 2: Build CNN Model
# ===============================

def build_cnn():

    model = Sequential([

        Conv2D(32, (3,3), activation='relu',
        input_shape=(28,28,1)),

        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),

        MaxPooling2D(2,2),

        Flatten(),

        Dense(128, activation='relu'),

        Dense(10, activation='softmax')

    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ===============================
# Step 3: Train Model
# ===============================

def train_model():

    x_train, y_train, x_test, y_test = load_and_preprocess()

    model = build_cnn()

    model.fit(
        x_train,
        y_train,
        epochs=5,
        validation_data=(x_test, y_test)
    )

    # Save model
    model.save("models/handwritten_cnn.h5")

    print("✅ Model saved successfully!")

    return model


# ===============================
# Step 4: Evaluate Model
# ===============================

def evaluate_model():

    x_train, y_train, x_test, y_test = load_and_preprocess()

    model = load_model("models/handwritten_cnn.h5")

    pred = model.predict(x_test)

    pred_labels = np.argmax(pred, axis=1)

    print("\n📊 Classification Report:\n")

    print(classification_report(
        y_test,
        pred_labels
    ))


# ===============================
# Step 5: Predict Custom Image
# ===============================

def predict_image(image_path):

    model = load_model("models/handwritten_cnn.h5")
    model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

    img = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    img = cv2.resize(img, (28,28))

    # Invert colors
    img = cv2.bitwise_not(img)

    img = img.astype("float32") / 255.0

    img = img.reshape(1,28,28,1)

    pred = model.predict(img)

    print("\n Predicted Digit:",np.argmax(pred))
    print("\n")


# ===============================
# Main Execution
# ===============================

if __name__ == "__main__":

    print("\n===== Training Model =====")
    train_model()

    print("\n===== Evaluating Model =====")
    evaluate_model()

    print("\n===== Predicting Image =====")

    # Change image name if needed
    predict_image("1.jpeg")