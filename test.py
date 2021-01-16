import time
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
# import commonfunctions
from skimage import io, color
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import minmax_scale
from commonfunctions import show_images, showHist
from sklearn import svm
from sklearn import metrics

def LBP(image,R):
    powers = [1,2,4,8,16,32,64,128]
    Xs = image.shape[0]
    Ys = image.shape[1]
    print(image.shape)
    start = time.time()
    newImage = np.zeros((Xs, Ys), np.uint8)
    end = time.time()
    print("np zeroes time",end-start)
    for z in range(0,Xs*Ys):
        i = z // Ys
        #print("i = ",i)
        j = z % Ys
        #get 8 points
        center = image[i][j]
        #upper left
        try:
            ul = 1 if center <= image[i-R][j-R] else 0
        except:
            ul = 0
        #print(ul)
        #upper
        try:
            u = 1 if center <= image[i-R][j] else 0
        except:
            u = 0
        #upper right
        try:
            ur = 1 if center <= image[i-R][j+R] else 0
        except:
            ur = 0
        #right
        try:
            r = 1 if center <= image[i][j+R] else 0
        except:
            r = 0
        #lower right
        try:
            lr = 1 if center <= image[i+R][j+R] else 0
        except:
            lr = 0
        #lower
        try:
            lw = 1 if center <= image[i+R][j] else 0
        except:
            lw = 0
        #lower left
        try: 
            ll = 1 if center <= image[i+R][j-R] else 0
        except:
            ll = 0
        #left
        try:
            l = 1 if center <= image[i][j-R] else 0
        except:
            l = 0
        #collect neighbours in an array
        neighbours = [ul,u,ur,r,lr,lw,ll,l]
        #print(neighbours)
        newCenter = 0
        for k in range(len(neighbours)):
            newCenter += neighbours[k]*powers[k]
        #print(newCenter)
        newImage[i][j] = newCenter
    return newImage
            
            
            
#img = cv.imread('Writerss/010/a01-020x.png')
img = cv.imread('line1.png')
#gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_img = img[:,:,0]
start = time.time()
readyOne = local_binary_pattern(gray_img, 8, 3)
readyOne = np.uint8(readyOne)
end = time.time()
print("ready one time",end-start)
start = time.time()
mine = LBP(gray_img,3)
end = time.time()
print("mine time",end-start)
#show_images([gray_img,readyOne,mine])

#show_images([gray_img,mine])