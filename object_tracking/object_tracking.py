import os
import sys
import cv2 as cv

import numpy as nm
import matplotlib.pyplot as plt
from base64 import b64encode

video_input_file_name = "race_car.mp4"

def drawRectangle(frame, bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

def displayRectangle(frame, bbox):
    plt.figure(figsize=(20, 10))
    frameCopy = frame.copy()
    drawRectangle(frameCopy, bbox)
    frameCopy = cv.cvtColor(frameCopy, cv.COLOR_BGR2RGB)
    plt.imshow(frameCopy)
    plt.axis("off")

def drawText(frame, txt, location, color=(50, 70, 50)):
    cv.putText(frame, txt, location, cv.FONT_HERSHEY_COMPLEX, 1, color, 3)

# Set up tracker
tracker_types = [
    "BOOSTING",
    "MIL",
    "KCF",
    "CSRT",
    "TLD",
    "MEDIANFLOW",
    "GOTURN",
    "MOSSE",
]

# Change the index to change the tracker type
tracker_type = tracker_types[6]

if tracker_type == "BOOSTING":
    tracker = cv.legacy.TrackerBoosting.create()
elif tracker_type == "MIL":
    tracker = cv.legacy.TrackerMIL.create()
elif tracker_type == "KCF":
    tracker = cv.TrackerKCF.create()
elif tracker_type == "CSRT":
    tracker = cv.TrackerCSRT.create()
elif tracker_type == "TLD":
    tracker = cv.legacy.TrackerTLD.create()
elif tracker_type == "MEDIANFLOW":
    tracker = cv.legacy.TrackerMedianFlow.create()
elif tracker_type == "GOTURN":
    tracker = cv.TrackerGOTURN.create()
else:
    tracker = cv.legacy.TrackerMOSSE.create()

video = cv.VideoCapture(video_input_file_name)
ok, frame = video.read()

if not video.isOpened():
    print("Could not open video")
    sys.exit()

width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

video_output_file_name = "race_car-" + tracker_type + ".mp4"
video_out = cv.VideoWriter(video_output_file_name, cv.VideoWriter_fourcc(*"XVID"), 10, (width, height))


bbox = (1300, 405, 106, 120)
displayRectangle(frame, bbox)
ok = tracker.init(frame, bbox)

while True:
    timer = cv.getTickCount()


    ok, frame = video.read()
    if not ok:
        break

    ok, bbox = tracker.update(frame)

    fps = cv.getTickFrequency() / (cv.getTickCount() - timer)

    if ok:
        drawRectangle(frame, bbox)
    else:
        drawText(frame, "tracking failure detected", (80, 40), (0, 0, 0))


    drawText(frame, tracker_type + " Tracker", (80, 60))
    drawText(frame, "FPS : " + str(int(fps)), (80, 100))
    video_out.write(frame)

video.release()
video_out.release()

