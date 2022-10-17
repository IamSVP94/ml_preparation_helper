'''
https://learnopencv.com/multitracker-multiple-object-tracking-using-opencv-c-python/ - Источник
https://learnopencv.com/object-tracking-using-opencv-cpp-python
'''
from __future__ import print_function
import sys
import cv2
from random import randint

# Step 1: Create a Single Object Tracker
import matplotlib.pyplot as plt

trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT', 'Siam']

# Set video to load
# videoPath = "/home/vid/Downloads/datasets/outC.mp4"
# videoPath = "/home/vid/hdd/file/project/RUS_AGRO/video/mini.mp4"
# videoPath = "/home/vid/hdd/file/project/RUS_AGRO/video/mini2.mp4"
videoPath = "/home/vid/hdd/projects/PycharmProjects/FACEID_VMX/temp/123.mp4"
# Specify the tracker type
trackerType = 'BOOSTING'  # from trackerTypes
# trackerType = 'CSRT'


def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.legacy.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.legacy.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.legacy.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.legacy.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.legacy.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.legacy.TrackerCSRT_create()
    elif trackerType == trackerTypes[5]:  # do not work
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[8]:  # do not work
        tracker = cv2.TrackerDaSiamRPN_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)

    return tracker


# Step 2: Read First Frame of a Video =========================


# Create a video capture object to read videos
cap = cv2.VideoCapture(videoPath)

# Read first frame
success, frame = cap.read()
# quit if unable to read the video file
if not success:
    print('Failed to read video')
    exit()
frame = cv2.resize(frame, (1440, 810), interpolation=cv2.INTER_LINEAR)  # resize

# Step 3: Locate Objects in the First Frame ===================

## Select boxes
bboxes = []
colors = []

# OpenCV's selectROI function doesn't work for selecting multiple objects in Python
# So we will call this function in a loop till we are done selecting all objects
while True:
    # draw bounding boxes over objects
    # selectROI's default behaviour is to draw box starting from the center
    # when fromCenter is set to false, you can draw box starting from top left corner
    bbox = cv2.selectROI('MultiTracker', frame)
    bboxes.append(bbox)
    colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    print("Press q to quit selecting boxes and start tracking")
    print("Press any other key to select next object")
    k = cv2.waitKey(0) & 0xFF
    if (k == 113):  # q is pressed
        break

print('Selected bounding boxes {}'.format(bboxes))

# Step 4: Initialize the MultiTracker =========================

# Create MultiTracker object
multiTracker = cv2.legacy.MultiTracker_create()

# Initialize MultiTracker
for bbox in bboxes:
    multiTracker.add(createTrackerByName(trackerType), frame, bbox)

# Step 4: Update MultiTracker & Display Results ===============
# Process video and track objects
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.resize(frame, (1440, 810), interpolation=cv2.INTER_LINEAR)  # resize

    # get updated location of objects in subsequent frames
    success, boxes = multiTracker.update(frame)

    # draw tracked objects
    for i, newbox in enumerate(boxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

    # show frame
    cv2.imshow('MultiTracker', frame)

    # quit on ESC button
    if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
        break
