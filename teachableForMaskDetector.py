import tensorflow.keras
import numpy as np
import cv2

# disable scientific notation for clarity
np.set_printoptions(suppress=False)

# load the model
model = tensorflow.keras.models.load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

img = cv2.imread('without2.jpg')
img = cv2.resize(img, (224, 224))

imageArray = np.asarray(img)

# normalize the image
normalizedImage = (imageArray.astype(np.float32)/127.0) - 1
data[0] = normalizedImage

# prediction
prediction = model.predict(data)
print(prediction)

for i in prediction:
    if i[0]>i[1]:
        text = "mask"
    else:
        text = "no mask"

    print(text)
    img = cv2.resize(img, (500, 500))
    cv2.putText(img, text, (10,30), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
