import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import skimage.feature.texture as ski
import math
from sklearn.neighbors import KNeighborsClassifier
import time

#get the neighbours in a list by moving clockwise
def getNeighbours(pointCoord):
    neighboursCoord = []
    '''
    get the neighbours starting from the top-left clockwise
    '''
    #print(pointCoord)
    #append the top neighbours moving from left to right
    for j in range(0,3):
        neighboursCoord.append([pointCoord[0] - 1,j])
    #append the right neighbour
    neighboursCoord.append([pointCoord[0],pointCoord[1] + 1])
    #append the bottom neighbours moving from right to left
    for j in range(2,-1,-1):
        neighboursCoord.append([pointCoord[0] + 1,j])
    #append the left neighbour
    neighboursCoord.append([pointCoord[0],pointCoord[1] - 1])
    return neighboursCoord


def CSP_LP(neighbours,N,Threshold):
    #get the accumlating of csp_lp from the neighbours
    CSP_LP_SUM = 0
    #loop on all neighbours
    for i in range(0,int(N/2)):
        #substract each neighbour and its peer
        s = neighbours[i] - neighbours[i + int(N/2)]
        #check if s is greater than the given threshold
        if(s > Threshold):
            #if the condition is satisfied then add pow(2,i) to csp_lp sum
            CSP_LP_SUM += (2**i)
    return CSP_LP_SUM

def getFeatureVector(img):
    GLCM_Input = []
    startTime = time.time()
    imgRows = img.shape[0]
    imgCols = img.shape[1]
    for i in range(1,imgRows - 1):
        GLCM_Row = []
        for j in range(1,imgCols - 1):
            neighbours = []
            neighboursCoord = getNeighbours([i,j])
            for p in neighboursCoord:
                x = p[0]
                y = p[1]
                neighbours.append(img[x][y])
            CSP_LP_SUM = CSP_LP(neighbours,8,0.01)
            GLCM_Row.append(CSP_LP_SUM)
        GLCM_Input.append(GLCM_Row)
    
    GLCM_Matrix = ski.greycomatrix(GLCM_Input,distances=[2],angles=[0,math.pi/4,
                                                    math.pi/2,math.pi*(5/4)],levels=16)
    #endTime = time.time()
    #print("time = ",startTime-endTime)
    featureVector = []
    for k in range(GLCM_Matrix.shape[3]):
        for i in range(GLCM_Matrix.shape[0]):
            for j in range(GLCM_Matrix.shape[1]):
                featureVector.append(GLCM_Matrix[i][j][0][k])
    endTime = time.time()
    print("time = ",endTime - startTime)
    return featureVector

#read the image and convert to normalized grayscale
img = cv.imread('../formsE-H/e01-014.png')
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#print(img)
#print(img.shape)
lines = preprocessing(gray_img)
features = []
labels = []

start = time.time()
for line in lines:
    startTime = time.time()
    line = cv.cvtColor(line, cv.COLOR_BGR2GRAY)/255
    endTime = time.time()
    print(endTime-startTime)
    print(line.shape)
    lineFeature = getFeatureVector(line)
    features.append(lineFeature)
end = time.time()
print(end-start)

kNeighbours = KNeighborsClassifier(n_neighbors=3)