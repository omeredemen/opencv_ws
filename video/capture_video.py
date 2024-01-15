import numpy as np
import cv2 as cv

cap = cv.VideoCapture(2)

if not cap.isOpened:
    print("Cannot open camera")
    exit()

print("height: ", cap.get(cv.CAP_PROP_FRAME_HEIGHT))
print("width: ", cap.get(cv.CAP_PROP_FRAME_WIDTH))

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame")
        break
    
    # Our operations on the frame come here
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # Display the resulting frame
    cv.imshow("frame", hsv)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()