from scipy import fftpack
import time
import cv2 as cv
import numpy as np
def getDCT(image,n):
    image = np.float32(image)
    dst =  fftpack.dct(fftpack.dct( image, axis=0,type=2, norm='ortho' ),axis=1, type=2, norm='ortho')
    print(dst.shape)
    rows=dst.shape[0]
    columns=dst.shape[1]
    matrix = dst
    solution=[[] for i in range(rows+columns-1)]
    for i in range(rows):
        for j in range(columns):
            sum=i+j
            if(sum%2 ==0):
            #add at beginning
                solution[sum].insert(0,matrix[i][j])
            else:
                #add at end of the list
                solution[sum].append(matrix[i][j])
    sol = np.hstack(solution)
    print(len(sol))
    return sol[0:n+1]

img = cv.imread('line1.png')
gray_img = img[:,:,0]
print(gray_img.shape)
start = time.time()
mine = getDCT(gray_img,100)
end = time.time()
print("dst time",end-start)
print(mine)