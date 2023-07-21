# By  KJC
import pyfirmata
from pyfirmata import SERVO
import tensorflow.keras
import numpy as np
import cv2
import pyttsx3
import time
import os


# setting the board
board = pyfirmata.ArduinoNano('com7')
# board = pyfirmata.ArduinoMega('com4')


# pins
red = board.get_pin('d:8:o')
green = board.get_pin('d:10:o')

# servo setting
servo = board.get_pin('d:6:o')
board.digital[6].mode = SERVO


servo.write(0)
# set the speaker parameters
speaker = pyttsx3.init()
voices = speaker.getProperty('voices')
speaker.setProperty('voice', voices[1].id)
speaker.setProperty('rate', 100)


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
    normalizedImage = (imageArray.astype(np.float32) / 127.0) - 1
    data[0] = normalizedImage

    # prediction
    prediction = model.predict(data)
    # print(prediction)

    for on, off, none, back in prediction:
        pre = max(on, off, none)
        if float(on) > float(off) and float(on) > float(none) and float(on) > float(back):
            text = "mask"
            servo.write(100)
            this = "Welcome to Robotic Exhibition"
            red.write(0)
            green.write(1)
        elif float(off) > float(on) and float(off) > float(none) and float(off) > float(back):
            text = "no mask"
            this = "Please Wear your mask"
            servo.write(0)
            red.write(1)
            green.write(0)
        else:
            text = ''
            this = ""
            servo.write(0)
            red.write(0)
            green.write(0)
            pass

        # img = cv2.resize(img, (500, 500))
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('KJC Mask Scanner', frame)
    speak(this)
    exitKey = cv2.waitKey(1)
    if exitKey == ord('q') or exitKey == ord('Q'):
        break

    # time.sleep(2)

cam.release()
cv2.destroyAllWindows()
os.remove('scan.jpg')

