import time

import numpy as np
import cv2
import os
import face_recognition
from datetime import datetime
import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise

# global variables
bg = None

path = "trainingData"
images=[]
classNames=[]


myList = os.listdir(path)
print(myList)

# names = f.readlines()
for cls in myList:
    curImage = cv2.imread(f'{path}/{cls}')
    images.append(curImage)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)

def markAttendance(name):
    with open(f'attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList=[]
        for line in myDataList:
            entry = (line.split(','))[0]
            nameList.append(entry)
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        f.write(f'\n{name},{dtString}')


def findEncodings(imgs):
    encodeList=[]
    for img in imgs:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encoding = face_recognition.face_encodings(img)[0]
        encodeList.append(encoding)
    return encodeList


def run_avg(image, accumWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, accumWeight)


# ---------------------------------------------
# To segment the region of hand in the image
# ---------------------------------------------
def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    cnts, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


# --------------------------------------------------------------
# To count the number of fingers in the segmented hand region
# --------------------------------------------------------------
def count(thresholded, segmented):
    # find the convex hull of the segmented hand region
    chull = cv2.convexHull(segmented)

    # find the most extreme points in the convex hull
    extreme_top = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right = tuple(chull[chull[:, :, 0].argmax()][0])

    # find the center of the palm
    cX = int((extreme_left[0] + extreme_right[0]) / 2)
    cY = int((extreme_top[1] + extreme_bottom[1]) / 2)

    # find the maximum euclidean distance between the center of the palm
    # and the most extreme points of the convex hull
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    maximum_distance = distance[distance.argmax()]

    # calculate the radius of the circle with 80% of the max euclidean distance obtained
    radius = int(0.8 * maximum_distance)

    # find the circumference of the circle
    circumference = (2 * np.pi * radius)

    # take out the circular region of interest which has
    # the palm and the fingers
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")

    # draw the circular ROI
    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)

    # take bit-wise AND between thresholded hand using the circular ROI as the mask
    # which gives the cuts obtained using mask on the thresholded hand image
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # compute the contours in the circular ROI
    (cnts, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # initalize the finger count
    count = 0

    # loop through the contours found
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # increment the count of fingers only if -
        # 1. The contour region is not the wrist (bottom area)
        # 2. The number of points along the contour does not exceed
        #     25% of the circumference of the circular ROI
        if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            count += 1

    return count

encodeListKnown = findEncodings(images)
print("encoding complete")



cap = cv2.VideoCapture(0)

if __name__ == "__main__":
    # initialize accumulated weight
    accumWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590

    # initialize num of frames
    num_frames = 0

    # calibration indicator
    calibrated = False

    # keep looping, until interrupted
    while (True):
        _, img = cap.read()
        imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

        faceCurrFrame = face_recognition.face_locations(imgSmall)
        encodeCurFrame = face_recognition.face_encodings(imgSmall, faceCurrFrame)

        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurrFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            face_dist = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(face_dist)
            (grabbed, frame) = camera.read()
            user=''

            # resize the frame
            frame = imutils.resize(frame, width=700)
            transaction = False

            # flip the frame so that it is not the mirror view
            frame = cv2.flip(frame, 1)

            # clone the frame
            clone = frame.copy()
            # get the height and width of the frame
            (height, width) = frame.shape[:2]

            # get the ROI
            roi = frame[top:bottom, right:left]

            # convert the roi to grayscale and blur it
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            # to get the background, keep looking till a threshold is reached
            # so that our weighted average model gets calibrated

            if matches[matchIndex] and face_dist[matchIndex] < 0.6:
                name = classNames[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                cv2.rectangle(img, (x1 * 4, y1 * 4), (x2 * 4, y2 * 4), (0, 255, 0), 2)
                cv2.rectangle(img, (x1 * 4, y2 * 4 - 35), (x2 * 4, y2 * 4), (0, 255, 0), cv2.FILLED)
                cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(img, name, (x1 * 4 + 6, y2 * 4 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                user = name

            if num_frames < 30:
                run_avg(gray, accumWeight)
                if num_frames == 1:
                    print("[STATUS] please wait! calibrating...")
                elif num_frames == 29:
                    print("[STATUS] calibration successfull...")
            else:
                # segment the hand region
                hand = segment(gray)

                # check whether hand region is segmented
                if hand is not None:
                    # if yes, unpack the thresholded image and
                    # segmented region
                    (thresholded, segmented) = hand

                    # draw the segmented region and display the frame
                    cv2.drawContours(img, [segmented + (right, top)], -1, (0, 0, 255))

                    # count the number of fingers
                    fingers = count(thresholded, segmented)
                    if(fingers>=2):
                        cv2.putText(img, 'Transaction Successful', (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        markAttendance(user)
                        transaction = True
                        break
                        # camera.release()
                        # cv2.destroyAllWindows()
                    else:
                        cv2.putText(img, 'Waiting for user confirmation', (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                    2)
                    # show the thresholded image
                    # cv2.imshow("Thesholded", thresholded)

            # draw the segmented hand
            # cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

            # increment the number of frames
            num_frames += 1

        cv2.imshow("Webcam", img)
        cv2.waitKey(1)
        # get the current frame

        # display the frame with segmented hand
        # cv2.imshow("Video Feed", img)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q") or transaction==True:
            time.sleep(2)
            break

# free up memory
camera.release()
cv2.destroyAllWindows()


    # if cv2.waitKey(0) & 0xFF==ord('q'):
    #     break



# print('trainingData/Bill Gates.jpeg')

# imagePath = f'trainingData/{names[1]}'
# print(imagePath)
# print('trainingData/Bill Gates.jpeg')
# if imagePath=='trainingData/Bill Gates.jpeg':
#     print(imagePath)
# gates = cv2.imread(imagePath)
# print(gates)


# cv2.imshow('Gates',gates)
# cv2.waitKey(0)
# cap = cv2.VideoCapture(0)

# while True:
#     # _,img = cap.read()
#     # cv2.imshow('Webcam',img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break