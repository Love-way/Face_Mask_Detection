# By  KJC


import tensorflow.keras
import numpy as np
import cv2
import pyttsx3
import time


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
model = tensorflow.keras.models.load_model('aspr_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# set the camera
cam = cv2.VideoCapture(0)

while cam.isOpened():

    # setting the image framework
    check, frame = cam.read()

    cv2.imwrite('scan.jpg', frame)

    img = cv2.imread('scan.jpg')
    img = cv2.resize(img, (224, 224))

    imageArray = np.asarray(img)

    # normalize the image
    normalizedImage = (imageArray.astype(np.float32)/127.0) - 1
    data[0] = normalizedImage

    # prediction
    prediction = model.predict(data)
    print(prediction)

    for on, off, none, back in prediction:
        if float(on)>float(off) and float(on)>float(none) and float(on)>float(back):
            text = "mask"
            this = "Welcome to KJC"
        elif float(off)>float(on) and float(off)>float(none) and float(off)>float(back):
            text = "no mask"
            this = "Please Wear your mask"
        elif float(none)>float(on) and float(none)>float(off) and float(none)>float(back):
            text = "empty"
            this = ""
            pass
        elif float(back)>float(on) and float(back)>float(off) and float(back)>float(none):
            text = 'out'
            this = "Thanks for KJC"
            pass


        print(text)
        
        # img = cv2.resize(img, (500, 500))
        cv2.putText(img, text, (10,30), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0), 2)
        
    
    lanbda: cv2.imshow('KJC Mask Scanner', img)#; speak(this)
    exitKey = cv2.waitKey(1)
    if exitKey == ord('q') or exitKey == ord('Q'):
        break

    # time.sleep(2)

cam.release()
cv2.destroyAllWindows()

