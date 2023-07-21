# By  KJC


import tensorflow.keras
import numpy as np
import cv2
import pyttsx3


# set the speaker parameters
speaker = pyttsx3.init()
voices = speaker.getProperty('voices')
speaker.setProperty('voice', voices[1].id)
speaker.setProperty('rate', 200)


# set a speak function
def speak(arg):
    speaker.say(arg)
    speaker.runAndWait()

# disable scientific notation for clarity
np.set_printoptions(suppress=False)

# load the model
model = tensorflow.keras.models.load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# set the camera
cam = cv2.VideoCapture(0)

while cam.isOpened():

    # setting the image framework
    check, frame = cam.read()

    cv2.imwrite('scanImage.jpg', frame)

    img = cv2.imread('scanImage.jpg')
    img = cv2.resize(img, (224, 224))

    imageArray = np.asarray(img)

    # normalize the image
    normalizedImage = (imageArray.astype(np.float32)/127.0) - 1
    data[0] = normalizedImage

    # prediction
    prediction = model.predict(data)
    print(prediction)

    for on, off in prediction:
        if float(off)<float(on):
            text = "mask"
            this = "Welcome to NIIT openlabs"
        elif float(off)>float(on):
            text = "no mask"
            this = "Please Wear your mask"
        else:
            pass


        print(text)
        
        # img = cv2.resize(img, (500, 500))
        cv2.putText(img, text, (10,30), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0), 2)
        
    
    lanbda: cv2.imshow('KJC Mask Scan', img); speak(this)
    exitKey = cv2.waitKey(1)
    if exitKey == ord('q') or exitKey == ord('Q'):
        break

cam.release()
cv2.destroyAllWindows()

