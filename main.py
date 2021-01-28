import numpy as np
import cv2
import os
import face_recognition
from datetime import datetime

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
        if name not in nameList:
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

encodeListKnown = findEncodings(images)
print("encoding complete")

cap = cv2.VideoCapture(0)

while True:
    _,img = cap.read()
    imgSmall =  cv2.resize(img,(0,0),None,0.25,0.25)
    imgSmall = cv2.cvtColor(imgSmall,cv2.COLOR_BGR2RGB)

    faceCurrFrame = face_recognition.face_locations(imgSmall)
    encodeCurFrame = face_recognition.face_encodings(imgSmall,faceCurrFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame,faceCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        face_dist = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(face_dist)
        matchIndex = np.argmin(face_dist)
        if matches[matchIndex] and face_dist[matchIndex]<0.6:
            name= classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            cv2.rectangle(img,(x1*4,y1*4),(x2*4,y2*4),(0,255,0),2)
            cv2.rectangle(img, (x1*4, y2*4-35), (x2*4,y2*4), (0, 255, 0), cv2.FILLED)
            cv2.putText(img,name,(x1*4+6,y2*4-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            markAttendance(name)

    cv2.imshow("Webcam",img)
    cv2.waitKey(1)
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