'''
ECE 276A Robot Sensing and Estimation Project 1
Stop Sign Detection and Localization
dataOperatiions.py (define some basic functionalities for data operations such as create train data...)
Baoqian Wang
0202
'''
import numpy as np
import cv2
import os.path as path
from skimage.measure import label, regionprops

def createTrainData(startIndex,stopIndex):
    #Read images from startIndex-th image to stopIndex-th image
    doiData=np.zeros((1,3))
    nrData=np.zeros((1,3))

    #The folder contains images in both jpg format and png format, and the number is not continuous
    for i in range(startIndex,stopIndex):
        testPath='./trainset/%s.jpg' %i
        if (path.exists(testPath)):
            imageFile=testPath
        elif path.exists('./trainset/%s.png'):
            imageFile='./trainset/%s.png'
        else:
            continue
        #Read the mask image
        maskFile='./maskset/mask%s.jpg' %i
        mask=cv2.imread(maskFile)
        mask = np.mean(mask, 2)
        #Read the image
        image=cv2.imread(imageFile)
        roi=image[mask>=240] #Get the region of interest (stop sign region)
        nr=image[mask<240]   #Get the non-stop sign region
        doiData=np.concatenate((doiData,roi))
        nrData=np.concatenate((nrData,nr))
    doiData=np.delete(doiData,0,axis=0)
    nrData=np.delete(nrData,0,axis=0)

    #Add the label to the each pixel value, 1 for stop sign region, -1 for non-stop sign region
    pl=np.ones((doiData.shape[0],1))
    nl = -1*np.ones((nrData.shape[0], 1))

    doiData=np.concatenate((doiData,pl,pl),axis=1)
    nrData=np.concatenate((nrData,-nl,nl),axis=1)
    return doiData, nrData

def getGroundBoxes(filename):
    #Get the ground boxes using the mask image file
    image = cv2.imread(filename)
    labelImage = label(image, connectivity=1)
    boxes = []
    for region in regionprops(labelImage):
        box=region.bbox
        boxes.append(box)

def evaluateBoxes(groundBoxes,boxes,errorToleranceTest):
    #Evaluate the accucary of the predicted box based on the ground box
    score=0
    for box in boxes:
        for groundBox in groundBoxes:
            if(np.abs(box[0]-groundBox[0])<=errorToleranceTest and np.abs(box[1]-groundBox[1])<=errorToleranceTest and np.abs(box[2]-groundBox[2])<=errorToleranceTest):
                score+=1