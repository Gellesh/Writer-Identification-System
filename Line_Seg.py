import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
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
