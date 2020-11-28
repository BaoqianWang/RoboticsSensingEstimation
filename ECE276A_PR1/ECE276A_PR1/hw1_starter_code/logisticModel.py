'''
ECE 276A Robot Sensing and Estimation Project 1
Stop Sign Detection and Localization
logisticModel.py (define the class of logisticModel and its functionalities including train, predict...)
Baoqian Wang
0202
'''
import numpy as np
import cv2

class LogisticModel:
    def __init__(self,trainData,lr,xdimension,maxIteration,errorTolerance):
        self.maxIteration=maxIteration
        self.errorTolerance=errorTolerance
        self.trainData=trainData
        self.lr=lr
        self.xdimension=xdimension
        self.weights=np.zeros((self.xdimension,1))

    def sigmoid(self,x):
        return 1.0 / (1.0 + np.exp(-1.0*x))

    def loadWeights(self,filename):
        weightsLoad=np.load(filename)
        self.weights=np.reshape(weightsLoad,(self.xdimension,1))

    def predictSinglePixel(self, testX):
        # Only predict a single pixel value
        value = np.dot(self.weights.transpose(), testX.reshape((self.xdimension, 1)))
        if (value >= 0):
            result = 1
        else:
            result = -1
        return result

    def label2pixel(self, rawImage):
        # Generate segmented image from prediction results
        row = rawImage.shape[0]
        column = rawImage.shape[1]
        image = np.zeros((row, column, 3))
        for i in range(row):
            for j in range(column):
                if rawImage[i, j] == 1.0:
                    image[i, j, 0] = 255
        return image

    def generateSegmentImage(self, image):
        # Predict each pixel in an image
        # The Image is 3-dimensional row-column-pixel
        row = image.shape[0]
        column = image.shape[1]
        bias = np.ones((row, column, 1))
        image = np.concatenate((image, bias), axis=2)
        resultImage = np.zeros((row, column))
        for i in range(row):
            for j in range(column):
                resultImage[i, j] = self.predictSinglePixel(image[i, j, :])
        segmentImage = self.label2pixel(resultImage)
        segmentImage = segmentImage.astype(np.uint8)
        segmentImage = cv2.cvtColor(segmentImage, cv2.COLOR_BGR2GRAY)
        return segmentImage

    def segment_image(self, img):
        mask_img = self.generateSegmentImage(img)
        ret, binary = cv2.threshold(mask_img, 2, 255, cv2.THRESH_BINARY)
        mask_img = binary / 255
        return mask_img

    def get_bounding_box(self, img):
        testImage = self.generateSegmentImage(img)
        contours, hierarchy = cv2.findContours(testImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cont in contours:
            epsilon = 0.001 * cv2.arcLength(cont, True)
            approx = cv2.approxPolyDP(cont, epsilon, True)
            if len(approx) >= 8 and cv2.contourArea(cont) >= 200:
                x, y, w, h = cv2.boundingRect(approx)
                box = [x, (testImage.shape[0] - y), (x + w), (testImage.shape[0] - y - h)]
                boxes.append(box)
        return boxes

    def train(self):
        #Use the maximum likelihood function to estimate the weights
        X=self.trainData[:,:self.xdimension]
        y=self.trainData[:,self.xdimension]
        dataLength=self.trainData.shape[0]
        y = y.reshape((dataLength, 1))
        #Use the gradient descent algorithm to update the weights
        for i in range(self.maxIteration):
            firstTerm=y.reshape((dataLength,1))*(1-self.sigmoid(y*np.matmul(X,self.weights)))
            gradient=np.matmul(firstTerm.transpose(),X)
            print('Iteration number', i)
            print('Current weights', self.weights)
            self.weights=self.weights+self.lr*gradient.transpose()
            if (np.linalg.norm(self.lr*gradient)<=self.errorTolerance):
                break

