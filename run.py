import matplotlib.pyplot as plt
import os
import time
from Classifiers import *
from featureExtraction import *
import sys 


""" contains helper functions to test accuracy and sample test set"""

def testing(clf,features,ids):
    trainF = np.array(features)
    y_pred = clf.predict(trainF)
    # print(y_pred)
    # print("Most frequent value in the above array:") 
    # print(np.bincount(y_pred).argmax())
    # print("Accuracy:",metrics.accuracy_score(trainLabels, y_pred),"\n")
    return 1 if np.bincount(y_pred).argmax() == ids[0] else 0

def run_tests(num,return_features,return_Labels):

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

    ids = [1,1,2,2,3,3,-1]
    #read all images
    pics = []
    for i in range(len(picsPath)):
        img = cv.imread(picsPath[i])
        pics.append(img)
    testImage =  cv.imread(testPath)
    pics.append(testImage)

    # start Time 
    start = time.time()

    #create data and train model
    #open 6 threads 
    # Create new threads
    threadList = [i for i in range (0,7)]
    threads = []
    for tName in threadList:
        f = []
        l = []
        thread = threading.Thread(target=get_features, args=(pics[tName],ids[tName],return_features,return_Labels,tName ))
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for t in threads:
        t.join()

    for i in range(6):
        features.extend(return_features[i])
        labels.extend(return_Labels[i])
  
    features = np.array(features)
    labels = np.array(labels)
  
    # clf = train_using_svm(features,labels)
    # clf = naive_Bayes(features,labels)
    clf = KNN(features,labels)
    
    #test model
    result = testing(clf,return_features[6],testId)
    # end time
    end = time.time()
    dur = end-start
    if result == 0:
        print("test case failed")
    return result ,dur

def get_result(num,return_features,return_Labels,rootDir):
    features = []
    labels = []
    testCase = num
    testDir = os.path.join(rootDir,testCase)
    picsPath = []
    for dirpath, dirnames, files in os.walk(testDir):
        if dirpath == testDir:
            testPath = os.path.join(dirpath,files[-1])
            continue
        for file in files:
            picsPath.append(os.path.join(dirpath,file))

    ids = [1,1,2,2,3,3,-1]
    #read all images
    pics = []
    for i in range(len(picsPath)):
        img = cv.imread(picsPath[i])
        pics.append(img)
    testImage =  cv.imread(testPath)
    pics.append(testImage)

    # start Time 
    start = time.time()

    #create data and train model
    #open 6 threads 
    # Create new threads
    threadList = [i for i in range (0,7)]
    threads = []
    for tName in threadList:
        f = []
        l = []
        thread = threading.Thread(target=get_features, args=(pics[tName],ids[tName],return_features,return_Labels,tName ))
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for t in threads:
        t.join()

    for i in range(6):
        features.extend(return_features[i])
        labels.extend(return_Labels[i])
  
    features = np.array(features)
    labels = np.array(labels)

    clf = KNN(features,labels)
    
    #test model
    trainF = np.array(return_features[6])
    y_pred = clf.predict(trainF)
    result = np.bincount(y_pred).argmax()
    # end time
    end = time.time()
    dur = end-start

    return result ,dur

def run_multiple(num ):

    # printing main program process id 
    # print("ID of main process: {}".format(os.getpid())) 
    manager = multiprocessing.Manager()
    return_features = manager.dict()
    return_Labels = manager.dict()
    
    testCasesNum = num
    skip = 0
    totalAcc = 0
    totalTime = 0
    for i in range(1 + skip,testCasesNum + 1 + skip):
        return_features = {}
        return_Labels = {}
        acc , ti = run_tests(str(i),return_features,return_Labels)
        totalAcc += acc 
        totalTime += ti

    print("Average accuracy ... = ",totalAcc/testCasesNum)
    print("Average time ... = ",totalTime/testCasesNum)

def test_set(folderPath = "TestSetSample"):

    dataPath = os.path.join(folderPath,"data")
    tests = os.listdir(dataPath)
    tests.sort(key=int)
    manager = multiprocessing.Manager()
    return_features = manager.dict()
    return_Labels = manager.dict()

    writer_file_object = open(os.path.join(folderPath,"results.txt"), 'a')
    time_file_object = open(os.path.join(folderPath,"time.txt"), 'a')


    for test in tests:
        writer , t = get_result(test,return_features,return_Labels,dataPath)

        # Add results to results.txt
        writer_file_object.write(str(writer) + "\n")

        # Add results to time.txt
        time_file_object.write(str(round(t,2)) + "\n")

        return_features = {}
        return_Labels = {}
        
    writer_file_object.close()
    time_file_object.close()


if __name__ == "__main__":

    argument = sys.argv

    if len(argument) == 1 :
        print("No path is entered please run again and enter TestSet path \n")
    else:
        test_set(argument[1])

    # runMultiple(500)
    #run_multiple(5)
    # test_set("Test Set Sample")

    


