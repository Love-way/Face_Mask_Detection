# By KJC

import cv2
import time

face_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(1)
while cap.isOpened():

    check, frame = cap.read()

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordonates = face_model.detectMultiScale(gray_img)
    
    if face_coordonates is not None:
        for x, y, w, h in face_coordonates:
            face_rectangles = cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)
            print('Please Wear you mask')
            pass
        pass
    else:
        print("Welcome")
        pass
    
    cv2.imshow("KJC face Detector", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
    time.sleep(1.5)

cap.release()

cv2.destroyAllWindows()

