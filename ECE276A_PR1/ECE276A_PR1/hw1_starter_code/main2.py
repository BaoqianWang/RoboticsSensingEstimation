'''
ECE276A WI20 HW1
Stop Sign Detector
'''

import os, cv2
from skimage.measure import label, regionprops
import numpy as np


class StopSignDetector():
    def __init__(self):
        '''
            Initilize your stop sign detector with the attributes you need,
            e.g., parameters of your classifier
        '''
        self.weights = np.array([[0.04232], [-0.10411], [0.06887], [-2.89652]])  # Initialize the weight
        self.xdimension = 4

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-1.0 * x))

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
        '''
            Obtain a segmented image using a color classifier,
            e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture,
            call other functions in this class if needed

            Inputs:
                img - original image
            Outputs:
                mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
        '''
        mask_img = self.generateSegmentImage(img)
        ret, binary = cv2.threshold(mask_img, 2, 255, cv2.THRESH_BINARY)
        mask_img = binary / 255
        return mask_img

    def get_bounding_box(self, img):
        '''
            Find the bounding box of the stop sign
            call other functions in this class if needed

            Inputs:
                img - original image
            Outputs:
                boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2]
                where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
                is from left to right in the image.

            Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
        '''
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


if __name__ == '__main__':
    folder = "testset"
    my_detector = StopSignDetector()
    for filename in os.listdir(folder):
        # read one test image
        img = cv2.imread(os.path.join(folder, filename))
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Display results:
        # (1) Segmented images
        mask_img = my_detector.segment_image(img)
        # (2) Get bounding boxes
        boxes = my_detector.get_bounding_box(img)

