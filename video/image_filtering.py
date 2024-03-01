import os
import sys
import numpy as np
import cv2 as cv

PREVIEW = 0
BLUR = 1
FEATURES = 2
CANNY = 3

feature_params = dict(maxCorners = 1000, qualityLevel = 0.2, 
                      minDistance = 15, blockSize = 10)

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

image_filter = PREVIEW
alive = True

window = "Camera Filters"

cv.namedWindow(window, cv.WINDOW_NORMAL)
result = None

cap = cv.VideoCapture(s)
if not cap.isOpened:
    print("Cannot open camera")
    exit()

while alive:
    has_frame, frame = cap.read()
    if not has_frame:
        break

    frame = cv.flip(frame, 1)

    if image_filter == PREVIEW:
        result = frame
    elif image_filter == CANNY:
        result = cv.Canny(frame, 100, 200)
    elif image_filter == BLUR:
        result = cv.blur(frame, (13, 13))
    elif image_filter == FEATURES:
        result = frame
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners = cv.goodFeaturesToTrack(frame_gray, **feature_params)
        if corners is not None:
            for x, y in np.float32(corners).reshape(-1, 2):
                cv.circle(result, (int(x), int(y)) , 10, (255, 0, 0), 1)

    cv.imshow(window, result)

    key = cv.waitKey(1)
    if key == ord("q"):
        alive = False
    elif key == ord("c"):
        image_filter = CANNY
    elif key == ord("b"):
        image_filter = BLUR
    elif key == ord("f"):
        image_filter = FEATURES
    elif key == ord("p"):
        image_filter = PREVIEW

cap.release()
cv.destroyAllWindows(window)

