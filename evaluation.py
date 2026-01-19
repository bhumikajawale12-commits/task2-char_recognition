from tensorflow.keras.models import load_model
from data_preprocessing import load_and_preprocess
from sklearn.metrics import classification_report
import numpy as np

x_train, y_train, x_test, y_test = load_and_preprocess()
model = load_model("../models/handwritten_cnn.h5")

pred = model.predict(x_test)
pred_labels = np.argmax(pred, axis=1)

print(classification_report(y_test, pred_labels))
