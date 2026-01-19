from data_preprocessing import load_and_preprocess
from model import build_cnn

x_train, y_train, x_test, y_test = load_and_preprocess()
model = build_cnn()

model.fit(x_train, y_train, epochs=5, validation_data=(x_test,y_test))

model.save("../models/handwritten_cnn.h5")
print("Model saved successfully!")
