'''
ECE 276A Robot Sensing and Estimation Project 1
Stop Sign Detection and Localization
main.py
Baoqian Wang
0202
'''
from hw1_starter_code.dataOperations import createTrainData
import numpy as np
from hw1_starter_code.logisticModel import LogisticModel
import cv2
import os
from hw1_starter_code.dataOperations import getGroundBoxes
from hw1_starter_code.dataOperations import evaluateBoxes

# Create training data set
doiData, nrData =createTrainData(1,50)  #Use images from 1 to 50
trainData=np.concatenate((doiData,nrData))

#Specify learning parameters
lr=0.002 #Learning rate
xdimension=4 #Dimension of X
maxIteration=30 #Number of iteration
errorToleranceTrain=0.001 #Error tolerance of weights
errorToleranceTest=100 #Error tolerance of bounding box


logModel=LogisticModel(trainData,lr,xdimension,maxIteration,errorToleranceTrain) #Create a Logistic Regression model
logModel.train() #Train the model
#logModel.loadWeights('modelWeights.npy') #The model can also load pretrained weights
folder = "testset"
for filename in os.listdir(folder):
    # read one test image
    img = cv2.imread(os.path.join(folder, filename))
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Display results:
    # (1) Segmented images
    mask_img =logModel.segment_image(img)
    # (2) Get bounding boxes
    boxes = logModel.get_bounding_box(img)
    groundBoxes=getGroundBoxes(filename)
    number=len(groundBoxes)
    score=evaluateBoxes(boxes,groundBoxes,errorToleranceTest) #Get the score to evaluate the accuracy of the bounding box
    for box in boxes:
        cv2.rectangle(img, (box[0], img.shape[0]-box[1]), (box[2], img.shape[1]-box[3]), (255, 255, 123), 2) #Draw the bounding box on the image
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()