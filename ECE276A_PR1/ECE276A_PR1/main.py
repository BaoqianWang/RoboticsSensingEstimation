from hw1_starter_code.dataOperations import createTrainData
import numpy as np
from hw1_starter_code.dataOperations import readImage
from hw1_starter_code.dataOperations import convertImage
from hw1_starter_code.logisticModel import LogisticModel
import cv2
doiData, nrData =createTrainData(1,10)
trainData=np.concatenate((doiData,nrData))
#ptrainData=trainData[:1000,:]
#ntrainData=trainData[-1000:-1]

#newTrainData=np.concatenate((ptrainData,ntrainData),axis=0)
lr=0.02
xdimension=4
maxIteration=20
errorTolerance=0.001
#(self,trainData,lr,xdimension,maxIteration,errorTolerance)
logModel=LogisticModel(trainData,lr,xdimension,maxIteration,errorTolerance)
logModel.train()

testImage=readImage(26)
# gray = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
# ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# cv2.imshow('image1',binary)
# cv2.waitKey(0)
predictImage=logModel.predictImage(testImage)
convertedImage=convertImage(predictImage)
convertedImage=convertedImage.astype(np.float32)
convertedImage = cv2.cvtColor(convertedImage, cv2.COLOR_BGR2GRAY)

canvas = np.zeros(convertedImage.shape, np.uint8)

convertedImage=convertedImage.astype(np.uint8)
ret, convertedImage = cv2.threshold(convertedImage, 2, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(convertedImage,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)











cnt = contours[0]
max_area = cv2.contourArea(cnt)

# for cont in contours:
#     if cv(cont) > max_area:
#         cnt = cont
        #max_area = cv2.contourArea(cont)

#epsilon = 0.1 * cv2.arcLength(cnt, True)
#approx = cv2.approxPolyDP(cnt, epsilon, True)
totalApprox=[]
for cont in contours:
    epsilon = 0.001*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cont,epsilon,True)
    totalApprox.append(approx)
    distance=[]
    if len(approx)>=4:
        center=np.mean(approx,axis=0)
        x, y, w, h = cv2.boundingRect(approx)
        cv2.rectangle(testImage, (x, y), (x + w, y + h), (255, 255, 123), 2)
        cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
        cv2.imshow('image2', testImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

center=np.mean(approx,axis=0)
distanceXY=np.square(approx-center)
distance=np.sum(distanceXY.transpose(), axis=0)
variance=np.var(distance)
# for cnt in contours:
#     cv2.drawContours(canvas, cnt, -1, (123, 255, 456), 3)
#     #convertedImage=convertedImage.astype(np.float32)
#     cv2.namedWindow('image1',cv2.WINDOW_NORMAL)
#     cv2.imshow('image1',canvas)
#     cv2.resizeWindow('image1', 600,600)
#     #cv2.resizeWindow('image2', 600,600)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()





cv2.rectangle(testImage,(x,y),(x+w,y+h),(255,255,123),2)
cv2.namedWindow('image2',cv2.WINDOW_NORMAL)
cv2.imshow('image2',testImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.namedWindow('image3',cv2.WINDOW_NORMAL)
cv2.imshow('image3',convertedImage)
cv2.waitKey(0)
cv2.destroyAllWindows()



#


#
# convertedImage=convertedImage.astype(np.float32)
# gray = cv2.cvtColor(convertedImage, cv2.COLOR_BGR2GRAY)
# cv2.imshow('image3',gray)
# cv2.resizeWindow('image3', 600,600)
# cv2.waitKey(0); cv2.destroyAllWindows()

#ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

#contours, hierarchy = cv2.findContours(convertedImage)
#cv2.drawContours(convertedImage, contours, -1, (0, 0, 255), 3)

#cv2.waitKey(0); cv2.destroyAllWindows();