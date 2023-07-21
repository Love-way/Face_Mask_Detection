import cv2

cam = cv2.VideoCapture(1)

i = 0
while cam.isOpened():

    check, frame = cam.read()

    cv2.imshow('KJC DataCollection', frame)

    key = cv2.waitKey(1)

    if key == ord('p'):
        cv2.imwrite('none{}.jpg'.format(i), frame)
        i += 1
        pass
    elif key == ord('q'):
        break

    pass

cv2.release()

cv2.destroyAllWindows()

