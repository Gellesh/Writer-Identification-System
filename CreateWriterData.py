import os
import shutil
import random

def generateWritersFolders(dataBases):
    """ This function takes array of dataBases and generate writers folders """

    # read metaData of the writers
    metaDataForm = "../forms.txt" 
    file = open(metaDataForm, "r")
    writer_dic = dict() 
    for line in file:
        if line.startswith("#"):
            continue
        try:
            x = line.split(" ")[:2]
            writer_dic[x[0]] = x[1]
        except:
            pass
        

    writer_dir = 'Writers' # The writers folder 
    try:
        os.mkdir(writer_dir)
    except FileExistsError as exc:
        pass
    # create Writers Folder and copy each image to writer number folder
    for db in dataBases:

        training_dir = db   # dataset folder
        entries = os.listdir(training_dir)
    
        for entry in entries:
            image_name = entry.split(".")[:1]
            writer_id = writer_dic[image_name[0]]
            
            try:
                os.mkdir(os.path.join(writer_dir,writer_id))
            except FileExistsError as exc:
                pass
            src = os.path.join(training_dir, entry)
            dst = os.path.join(writer_dir, writer_id, entry)
            shutil.copy(src, dst)



def getStats(num):
    """Walking a directory tree and printing the names of the directories and files bigeer than num"""
    root = "Writers"
    threCount = 0
    for dirpath, dirnames, files in os.walk(os.path.join(root)):
        if dirpath == root :
            continue
        count = 0
        for file_name in files:
            count += 1
        if count > num:
            threCount += 1 
            print("Writer {} has {} Imgs \n".format(dirpath,count))
    print(threCount)


def deleteWriters(num):
    """Walking a directory tree and delete directories of writers that have imgs less than num"""
    root = "Writers"
    threCount = 0
    for dirpath, dirnames, files in os.walk(os.path.join(root)):
        if dirpath == root :
            continue
    #     print(f'Found directory: {dirpath}')
        count = 0
        for file_name in files:
            count += 1
        if count < 2:
            threCount += 1 
            print("Writer {} has {} Imgs \n".format(dirpath,count))
            try:
                shutil.rmtree(dirpath)
            except OSError as e:
                print(f'Error: {trash_dir} : {e.strerror}')

    print(threCount)


# 1 for windows 
# 0 for linux

def GenerateTestCases(numTest = 1 , type = 1):
    " Generate test cases equal to numTest paramter and for linux users pass type = 0 "
      
    root = "Writers"
    Writers = [] # writers folders contains more than 3 photos
    threCount = 0
    for dirpath, dirnames, files in os.walk(os.path.join(root)):
        if dirpath == root :
            continue
    #     print(f'Found directory: {dirpath}')
        count = 0
        for file_name in files:
            count += 1
        if count > 2:
            threCount += 1 
            if type == 1:
                Writers.append(dirpath.split("\\")[1])
            else:
                 Writers.append(dirpath.split("/")[1])

    rootDir = "TestCases"
    try:
        os.mkdir(rootDir)
    except FileExistsError as exc:
        pass

    file1 = open(os.path.join(rootDir ,"results.txt"), "w")
    for i in range(1, 1 + numTest):    
        random.shuffle(Writers)
        selectedWriters = random.sample(Writers,3)
        path = os.path.join(rootDir,str(i))
        try:
            os.mkdir(path)
        except FileExistsError as exc:
            pass
        
        for k in range(1,4):
            try:
                os.mkdir(os.path.join(path,str(k)))
            except FileExistsError as exc:
                pass
            
        testImgs = []
         
        for num, writer in enumerate(selectedWriters,1):
            # create writer folder
            try:
                os.mkdir(os.path.join(path,str(num)))
                
            except FileExistsError as exc:
                pass
            # set src and des folder paths and get writer Images
            testCasePath = os.path.join(path,str(num))
            oldImagePath = os.path.join(root,writer)
            WriterImgs = os.listdir(oldImagePath)
            # sample 3 random Images:
            WriterImgs = random.sample(WriterImgs,3)
            # copy first 2 imgs in writer folder
            for img in WriterImgs[:2]:
                src = os.path.join(oldImagePath, img)
                dst = os.path.join(testCasePath, img)
                shutil.copy(src, dst)
            testImgs.append((num,os.path.join(oldImagePath,WriterImgs[2])))
            
            # print("Writer {} imgs {} copied to {} from {}".format(writer,WriterImgs[:2],testCasePath,oldImagePath))
        # get a third random img as testcase
        testImg = random.choice(testImgs)
        src = testImg[1]
        dst = os.path.join(path, "test." +testImg[1].split(".")[1])
        shutil.copy(src, dst)
        # print("Test img from writer {} at dest {} \n\n".format(testImg,dst))
        file1.write("For TestCase {} the test Img belong to writer -> {}\n\n".format(i,testImg[0]))

    file1.close()     
    

# make writes directory 

# dataBases = ["formsA-D"]
#dataBases = ["../formsA-D","../formsE-H"]
#generateWritersFolders(dataBases)

GenerateTestCases(500) # generate 5 test Cases    

# deleteWriters(2)  # to delete writes having 1 img only