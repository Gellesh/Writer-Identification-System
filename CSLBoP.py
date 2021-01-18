import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import skimage.feature.texture as ski
import math
from sklearn.neighbors import KNeighborsClassifier
import time
from Line_Seg import *
import os
from sklearn.preprocessing import minmax_scale
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
    GLCM_Input = [] #to store the CSLBP values
    startTime = time.time() #start timer
    #get image shapes
    imgRows = img.shape[0] 
    imgCols = img.shape[1]
    #loop on all pixels starting from the 1 row till the row before the last one and same for columns
    for i in range(1,imgRows - 1):
        GLCM_Row = []
        for j in range(1,imgCols - 1):
            neighbours = []
            neighboursCoord = getNeighbours([i,j]) #get the pixels' neighbours coordinates
            for p in neighboursCoord:
                x = p[0]
                y = p[1]
                neighbours.append(img[x][y]) #get those neighbours from their coordinates
            CSP_LP_SUM = CSP_LP(neighbours,8,0.01) #get the CSLBP value from those neighours
            GLCM_Row.append(CSP_LP_SUM) #append the CSP_LP_SUM to the row of the GLCM list
        GLCM_Input.append(GLCM_Row) #append the GLCM_ROW to the GLCM_Input
    
    #initialize hte GLCM matrix with four directino = 0,45,90,135 with distance = 1 and levels = 16
    GLCM_Matrix = ski.greycomatrix(GLCM_Input,distances=[1],angles=[0,math.pi/4,
                                                    math.pi/2,math.pi*(5/4)],levels=16)
    #get all the features by appending the 4 matrices into single feature vector
    allFeatures = []
    featureVector = []
    for k in range(GLCM_Matrix.shape[3]):
        for i in range(GLCM_Matrix.shape[0]):
            for j in range(GLCM_Matrix.shape[1]):
                featureVector.append(GLCM_Matrix[i][j][0][k])
    return featureVector

def runTests(num):

    features = []
    labels = []
    rootDir = "TestCases"
    testCase = num
    testDir = os.path.join(rootDir,testCase)
    picsPath = []
    testPath =""
    for dirpath, dirnames, files in os.walk(testDir):
        if dirpath == testDir:
            testPath = os.path.join(dirpath,files[-1])
            continue
        for file in files:
            picsPath.append(os.path.join(dirpath,file))

    testId = []
    with open(os.path.join(rootDir,"results.txt")) as fp: 
        Lines = fp.readlines() 
        for line in Lines: 
            if "TestCase " + testCase in line:
                testId.append(int(line[len(line)-2]))
                break

    print("Test case num {} belongs to writer {} ".format(num,testId))
    
    ids = [1,1,2,2,3,3]
    #read all images
    pics = []
    features = []
    labels = []
    
    totatTime = 0
    for i in range(len(picsPath)):
        img = cv.imread(picsPath[i]) #read the image number i
        startTime = time.time()
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #convert it to gray scale
        lines,bin_lines = preprocessing(gray_img) #get the grayscaled lines
        for line in lines:
            line = minmax_scale(line) #normalize the line
            lineFeature = getFeatureVector(line) #get the feature of the line
            features.append(minmax_scale(lineFeature)) #append the feature of the line
            labels.append(ids[i]) #append the label of the line
        endTime = time.time()
        totatTime += endTime - startTime
    testImage =  cv.imread(testPath)
    
    startTime = time.time()
    gray_img = cv.cvtColor(testImage, cv.COLOR_BGR2GRAY)
    lines,bin_lines = preprocessing(gray_img)
    
    '''
        get the features of the test image lines
    '''
    testFeatures = []
    for line in lines:
        line = minmax_scale(line)
        lineFeature = getFeatureVector(line)
        testFeatures.append(minmax_scale(lineFeature))
    KNN = KNeighborsClassifier(n_neighbors=10)
    KNN.fit(features,labels)
    predicted_Label = KNN.predict(testFeatures)
    endTime = time.time()
    totatTime += endTime - startTime
    print("total time = ",totatTime)
    if np.bincount(predicted_Label).argmax() == testId[0]:
        return 1
    else:
        return 0
    

accuracy = 0
for iter in range(50):
    accuracy += runTests(str(iter+1))
    print(accuracy/50)

print(accuracy)
print(accuracy/50)