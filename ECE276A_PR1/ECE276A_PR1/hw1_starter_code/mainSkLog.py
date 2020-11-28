from hw1_starter_code.dataOperations import createTrainData
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
doiData, nrData =createTrainData(1,70)
trainData=np.concatenate((doiData,nrData))


lr=0.002
xdimension=4
maxIteration=30
errorTolerance=0.001

samples=trainData[:,:4]
target=trainData[:,4]
samples, target = shuffle(samples,target, random_state=0)
classifier=LogisticRegression()
classifier.fit(samples,target).score(samples, target)


# logModel=LogisticModel(trainData,lr,xdimension,maxIteration,errorTolerance)
# #logModel.train()
# logModel.loadWeights()
# ratio0=0.5
# ratio1=0.5
#
# testImage=readImage(53)
# # gray = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
# # ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# # cv2.imshow('image1',binary)
# # cv2.waitKey(0)
# predictImage=logModel.predictImage(testImage)
# convertedImage=convertImage(predictImage)
# convertedImage=convertedImage.astype(np.float32)
# convertedImage = cv2.cvtColor(convertedImage, cv2.COLOR_BGR2GRAY)
#
#
# #convertedImage=convertedImage.astype(np.uint8)
# ret, binary = cv2.threshold(convertedImage, 2, 255, cv2.THRESH_BINARY)
# binary=binary/255
# cv2.imshow('image2', binary)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#contours, hierarchy = cv2.findContours(convertedImage,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

'''
#cnt = contours[0]
max_area = 0
FirstContours=[]
for cont in contours:
    epsilon = 0.01 * cv2.arcLength(cont, True)
    #perimeter = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, epsilon, True)
    if len(approx)>=4 and cv2.contourArea(cont)>=100:
        FirstContours.append(cont)



#epsilon = 0.1 * cv2.arcLength(cnt, True)
#approx = cv2.approxPolyDP(cnt, epsilon, True)
totalApprox=[]
totalVariance=[]

while(1):
    if(len(FirstContours) == 0):
        break
    maxCnt=getMaxAreaCnt(FirstContours)
    epsilon1 = 0.001 * cv2.arcLength(maxCnt, True)
    approx1 = cv2.approxPolyDP(maxCnt, epsilon1, True)
    x1, y1, w1, h1 = cv2.boundingRect(approx1)
    numRect=0
    index=[]
    tempIndex=-1
    for cont in FirstContours:
        tempIndex=tempIndex+1
        epsilon = 0.001*cv2.arcLength(cont,True)
        approx = cv2.approxPolyDP(cont,epsilon,True)
        x, y, w, h = cv2.boundingRect(approx)
        isInside=insideRect(x1,y1,w1,h1,x,y,w,h)
        numRect=numRect+isInside
        if(isInside):
            index.append(tempIndex)
    index.sort(reverse=True)

    if numRect>=2:
        cv2.rectangle(testImage, (x1, y1), (x1 + w1, y1 + h1), (255, 255, 123), 2)
        cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
        cv2.imshow('image2', testImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        for indexV in index:
            del FirstContours[indexV]
            #FirstContours.remove(FirstContours[indexV])
        continue
        #break
    else:
        FirstContours.remove(maxCnt)
'''
