import cv2 as cv
import numpy as np
from skimage import io, color
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import minmax_scale
import threading
import multiprocessing 
from Line_Seg import preprocessing

""" Extract features form images using LBP """

def LBP_feature_extraction(lines,bin_lines, features, labels, label):
  
    R = 3
    P = 8
    for i in range(len(lines)):
        #rgb to grat scale
        # grayImage = cv.cvtColor(lines[i], cv.COLOR_BGR2GRAY)
        #calc PBL
        LBPImage = local_binary_pattern(lines[i], P, R)
        #change format for histogram function
        LBPImage = np.uint8(LBPImage)     
        #calculate the histogram
        LBPHist = cv.calcHist([LBPImage],[0],bin_lines[i],[256],[0,256])
        #normalize the histogram 0-1
        normalizedHist = minmax_scale(LBPHist)        
        features.append(normalizedHist[:,0])
        labels.append(label)
      
def get_features(pic,id, return_features = None,return_Labels =None,num = None):
    features = []
    labels = []
    gray_img = cv.cvtColor(pic, cv.COLOR_BGR2GRAY)
    lines,bin_lines = preprocessing(gray_img)
    LBP_feature_extraction(lines,bin_lines, features, labels, id)
    
    if return_features is not None:
        return_features[num] = features
        return_Labels[num] = labels
    