import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("models/handwritten_cnn.h5")

def predict_image(image_path):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (28,28))

    img = cv2.bitwise_not(img)

    img = img.astype("float32") / 255.0

    img = img.reshape(1,28,28,1)

    pred = model.predict(img)
    print("Predicted Digit:", np.argmax(pred))

predict_image("1.jpeg")

