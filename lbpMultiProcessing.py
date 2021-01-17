import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import time
# import commonfunctions
from skimage import io, color
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import minmax_scale
from commonfunctions import show_images, showHist
from sklearn import svm
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier 
import multiprocessing 

counterww = 0

def get_paragraph(gray_img, bin_img):

    height, width = gray_img.shape

    contours, hierarchy = cv.findContours(bin_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    threshold_width = 1500


    up, down, left, right = 0, height - 1, 0, width - 1

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)

        if w < threshold_width:
            continue

        if y < height / 2:
            if y > up:
                up = y
        else:
            down = y

    th = 0
    bin_img = bin_img[up:down + 1, left:right + 1]
    gray_img = gray_img[up:down + 1, left:right + 1]
    # Apply erosion to remove noise and dots.
    kernel = np.ones((3, 3), np.uint8)
    bin_img = cv.erode(bin_img, kernel, iterations=3)
    pts = np.nonzero(bin_img)
    x_min, y_min, x_max, y_max = min(pts[0]), min(pts[1]), max(pts[0]), max(pts[1])
    bin_img = bin_img[x_min-th:x_max+th, y_min-th:y_max+th]
    gray_img = gray_img[x_min-th:x_max+th, y_min-th:y_max+th]
    # Return the handwritten paragraph
    return gray_img, bin_img

def preprocessing(gray_img):
    gray_img = cv.GaussianBlur(gray_img, (5, 5), 0)
    ## (2) threshold
    thresh, bin_img = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
    gray_img, bin_img = get_paragraph(gray_img, bin_img)
    thresh, bin_img = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
#     plt.imshow(bin_img)
    hist = cv.reduce(bin_img,1, cv.REDUCE_AVG).reshape(-1)
#     for i in range(len(hist)):
#         print(i, hist[i])
    th = 2
    H,W = bin_img.shape[:2]
    uppers = []
    lowers = []
    if hist[0] > th:
        uppers.append(0)
    
     
    for y in range(H-1):
        if hist[y]<=th and hist[y+1]>th:
            uppers.append(y)
     
    for y in range(H-1):
        if hist[y]>th and hist[y+1]<=th:
            lowers.append(y)
            
    if hist[len(hist)-1] > th:
        lowers.append(len(hist)-1)

#     img = cv.cvtColor(gray_img, cv.COLOR_GRAY2BGR)
    
    lines = []
    bin_lines = []
    temp_uppers = uppers.copy()
    temp_lowers = lowers.copy()
    for i in range(len(uppers)):
        if lowers[i] - uppers[i] > 50:
            lines.append(gray_img[uppers[i]:lowers[i], :])
            bin_lines.append(bin_img[uppers[i]:lowers[i], :])
        else:
            temp_uppers.remove(uppers[i])
            temp_lowers.remove(lowers[i])
#     print(temp_uppers)
#     print(temp_lowers)

    count = 1
#     for l in bin_lines:
#         cv.imwrite("line" + str(count) + ".png", l)
#         count+=1
    
    return lines, bin_lines

lbpTime = 0
lbpHist = 0
normalizeTime = 0

def LBP_feature_extraction(lines,bin_lines, features, labels, label):
    global counterww
    global lbpTime 
    global lbpHist 
    global normalizeTime

    R = 3
    P = 8
    for i in range(len(lines)):
        #rgb to grat scale
        # grayImage = cv.cvtColor(lines[i], cv.COLOR_BGR2GRAY)
        #calc PBL
        start = time.time()
        LBPImage = local_binary_pattern(lines[i], P, R)
        lbpTime += time.time() - start
        #change format for histogram function
        LBPImage = np.uint8(LBPImage)     
        #calculate the histogram
        start = time.time()
        LBPHist = cv.calcHist([LBPImage],[0],bin_lines[i],[256],[0,256])
        lbpHist += time.time() - start
        #normalize the histogram 0-1
        start = time.time()
        normalizedHist = minmax_scale(LBPHist)
        normalizeTime += time.time() - start
        
        #normalizedHist = LBPHist/np.mean(LBPHist)
        # print(normalizedHist[:,0][:3])
        # showHist(normalizedHist)
        features.append(normalizedHist[:,0])
        labels.append(label)
        #plot histogram
        # plt.hist(normalizedHist, bins=256)
        # name = "Image_"+str(counterww) + ".png"
        # plt.savefig(os.path.join("Images",name))
        # counterww =  counterww + 1
        # plt.show()

def get_features(pics,features,labels,ids, return_features = None,return_Labels =None):
    for i in range(len(pics)):
        gray_img = cv.cvtColor(pics[i], cv.COLOR_BGR2GRAY)
        lines,bin_lines = preprocessing(gray_img)
        LBP_feature_extraction(lines,bin_lines, features, labels, ids[i])
    # print(features)
    if return_features is not None:
        return_features.extend(features)
        return_Labels.extend(labels)
    
    # print("ID of process running: {}  with features {}".format(os.getpid() , len(features))) 

def train_using_svm(features,labels):
    clf = svm.SVC(kernel='linear'  ,C=5.0) # Linear Kernel
    clf.fit(features, labels)
    return clf


def naive_Bayes(features,labels):
    gnb = GaussianNB().fit(features, labels) 
    return gnb
 

def KNN(features,labels):
    knn = KNeighborsClassifier(n_neighbors = 5).fit(features, labels) 
    return knn


def testing(clf,testImage,ids):
    trainF = []
    trainLabels = []
    testPic = [testImage]
    get_features(testPic,trainF,trainLabels,ids)
    trainF = np.array(trainF)
    trainLabels = np.array(trainLabels)
    y_pred = clf.predict(trainF)
    print(y_pred)
    # print("Most frequent value in the above array:") 
    print(np.bincount(y_pred).argmax())
    # print("Accuracy:",metrics.accuracy_score(trainLabels, y_pred),"\n")
    return 1 if np.bincount(y_pred).argmax() == ids[0] else 0

def runTests(num,return_features,return_Labels):

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

    # print(picsPath)
    # print(testPath)
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
    for i in range(len(picsPath)):
        img = cv.imread(picsPath[i])
        pics.append(img)
    testImage =  cv.imread(testPath)

    # start Time 
    start = time.time()

    #create data and train model
    #open 6 processes
    p0 = multiprocessing.Process(target=get_features, args=([pics[0]],f0,l0,[ids[0]],return_features,return_Labels ))
    p1 = multiprocessing.Process(target=get_features, args=([pics[1]],f1,l1,[ids[1]],return_features,return_Labels ))
    p2 = multiprocessing.Process(target=get_features, args=([pics[2]],f2,l2,[ids[2]],return_features,return_Labels ))
    p3 = multiprocessing.Process(target=get_features, args=([pics[3]],f3,l3,[ids[3]],return_features,return_Labels ))
    p4 = multiprocessing.Process(target=get_features, args=([pics[4]],f4,l4,[ids[4]],return_features,return_Labels ))
    p5 = multiprocessing.Process(target=get_features, args=([pics[5]],f5,l5,[ids[5]],return_features,return_Labels ))
    p0.start()
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p0.join()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    
    features = return_features
    labels = return_Labels
    # print(features)
    # print("""""""""""""""""""""""""""""""""""")
    # print(labels)

    #get_features(pics,features,labels,ids)
    features = np.array(features)
    labels = np.array(labels)
    # print(features.shape , labels.shape)

    # clf = train_using_svm(features,labels)
    # clf = naive_Bayes(features,labels)
    clf = KNN(features,labels)
    
    #test model
    result = testing(clf,testImage,testId)
    # end time
    end = time.time()
    dur = end-start
    # print("test case took {} sec".format(dur))
    return result ,dur



if __name__ == "__main__": 
    # printing main program process id 
    print("ID of main process: {}".format(os.getpid())) 
    manager = multiprocessing.Manager()
    return_features = manager.list()
    return_Labels = manager.list()
    
    testCasesNum = 1
    totalAcc = 0
    totalTime = 0
    for i in range(1,testCasesNum + 1):
        acc , ti = runTests(str(i),return_features,return_Labels)
        totalAcc += acc 
        totalTime += ti

    print("Average accuracy ... = ",totalAcc/testCasesNum)
    print("Average time ... = ",totalTime/testCasesNum)

    # acc , ti = runTests(str(48))

    print(lbpTime)
    print(lbpHist)
    print(normalizeTime)

