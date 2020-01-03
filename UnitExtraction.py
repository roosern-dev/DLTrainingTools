import cv2
from matplotlib import pyplot as plt
import numpy as np
import os

def on_trackbar(val):
    ret, thr2 = cv2.threshold(imgGreen,val,255,cv2.THRESH_BINARY)
    cv2.imshow('window', thr2)
    

def on_trackbar_adaptive1(val):
    if (val % 2 == 1 ):
        global globalThreshVal1 
        globalThreshVal1 = val
        th1 = cv2.adaptiveThreshold(imgRed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, globalThreshVal1,globalThreshVal2)
        cv2.imshow('window', th1)

def on_trackbar_adaptive2(val):
    global globalThreshVal2 
    globalThreshVal2= val
    th1 = cv2.adaptiveThreshold(imgRed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, globalThreshVal1,globalThreshVal2)
    cv2.imshow('window', th1)

def on_trackbar_morph(val):
    global globalMorph
    globalMorph = np.ones((val,val),np.uint8)
    opening = cv2.morphologyEx(th1, cv2.MORPH_OPEN, globalMorph)
    cv2.imshow('window', opening)

def getCornerSquare(oriImage, centroidCol, centroidRow, size):
    
    startCol = int(centroidCol - size/2)
    startRow = int(centroidRow - size/2)
    endCol = int(centroidCol + size/2)
    endRow = int(centroidRow + size/2)

    image = oriImage[startRow:endRow, startCol:endCol]

    return image

def getCorners(oriImage, startCol, startRow, width, height, size):
    topLeft = getCornerSquare(oriImage, startCol, startRow, size)
    topRight = getCornerSquare(oriImage, startCol + width, startRow, size)
    bottomLeft = getCornerSquare(oriImage, startCol, startRow + height, size)
    bottomRight = getCornerSquare(oriImage, startCol + width, startRow + height, size)

    return topLeft, topRight, bottomLeft, bottomRight


def main():
    dir_path = r'E:\Estek-AIProject\INTEL-SOOP\RaspberryPiFolder\Images\Training\SavedImages'
    #main directory level
    dest_dir_path = r'E:\Estek-AIProject\INTEL-SOOP\RaspberryPiFolder\Images\Training\TrainingSet31_12_19\CornersExtracted64px'
    for filename in os.listdir(dir_path):
        #unit type level (XCC,BHP,LCC)
        file_path = os.path.join(dir_path, filename)

        if os.path.isdir(file_path):
            destFilePath = os.path.join(dest_dir_path, filename)
            for filename in os.listdir(file_path):
                #SOOP/GOOD level
                fileIndex = 0
                subDirPath = os.path.join(file_path, filename)

                if os.path.isdir(subDirPath):
                    category = str(filename)
                    targetSubDirPath = os.path.join(destFilePath, filename)
                    if not os.path.exists(targetSubDirPath):
                        os.makedirs(targetSubDirPath)
                    for image in os.listdir(subDirPath):
                        #individual images level
                        imgPath = os.path.join(subDirPath, image)
                        if os.path.isfile(imgPath):

                            globalThreshVal1 = 0
                            globalThreshVal2 = 0
                            ret = 0
                            thr2 = 0
                            readFlag = cv2.IMREAD_COLOR

                            img = cv2.imread(imgPath, readFlag)

                            imgOri = img.copy()
                            if (img is None):
                                print('no image')
                            imgBlue, imgGreen, imgRed = cv2.split(img)
                            #print('the image type is '+img.type())
                            cv2.namedWindow('window', cv2.WINDOW_NORMAL)
                            #cv2.imshow('window', img)
                            #cv2.waitKey(0)
                            #cv2.destroyAllWindows()

                            '''
                            cv2.createTrackbar('slider', 'window', 0, 255, on_trackbar)
                            on_trackbar(0)
                            cv2.waitKey(0)
                            cv2.destroyWindow('window')
                            '''

                            globalThreshVal = 80
                            ret, thr2 = cv2.threshold(imgRed, globalThreshVal,255,cv2.THRESH_BINARY)
                            #cv2.imshow('window', thr2)
                            #cv2.waitKey(0)

                            '''
                            cv2.createTrackbar('slider1', 'window', 0, 200, on_trackbar_adaptive1)
                            cv2.createTrackbar('slider2', 'window', 0, 100, on_trackbar_adaptive2)
                            cv2.waitKey(0)
                            cv2.destroyWindow('window')
                            '''

                            globalThreshVal1 = 199
                            globalThreshVal2 = 0
                            th1 = cv2.adaptiveThreshold(imgRed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, globalThreshVal1,globalThreshVal2)
                            #cv2.imshow('window',th1)
                            #cv2.waitKey(0)
                            '''
                            cv2.createTrackbar('morphControl','window',0,50, on_trackbar_morph)
                            cv2.waitKey(0)
                            cv2.destroyWindow('window')
                            '''
                            globalMorph = np.ones((10,10),np.uint8)
                            opening = cv2.morphologyEx(th1, cv2.MORPH_OPEN, globalMorph)
                            #cv2.imshow('window', opening)
                            #cv2.waitKey(0)

                            contours, hierarchy = cv2.findContours(opening,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            #cv2.drawContours(img, contours, -1, (0,0,255), 3)
                            #cv2.imshow('window', img)
                            #cv2.waitKey(0)

                            squareContours = []
                            for contour in contours:
                                epsilon = 0.1*cv2.arcLength(contour, True)
                                approx = cv2.approxPolyDP(contour, epsilon, True)
                                squareContours.append(approx)

                            squareContours4 = []
                            squareContours8 = []
                            boundingBoxOversize = 20

                            for contour in squareContours:
                                if contour.size == 4:
                                    squareContours4.append(contour)
                                elif contour.size == 8:
                                    x,y,w,h = cv2.boundingRect(contour)
                                    if (w>100 and h>100):
                                        #crop only the corners with a square of 32 x 32
                                        squareContours8.append(contour)
                                        startingX = x - boundingBoxOversize
                                        startingY = y - boundingBoxOversize
                                        endingX = x + w + boundingBoxOversize
                                        endingY = y + h + boundingBoxOversize

                                        startCor = (startingX, startingY)
                                        endCor = (endingX, endingY)

                                        cv2.rectangle(imgOri, startCor, endCor, (255,0,0))
                                        imgCrop = img[startingY:endingY, startingX:endingX]


                                        for cornerImage in getCorners(img, x, y, w, h, 64):
                                            if(cornerImage.size != 0):
                                                imgSavePath = targetSubDirPath + "\\" + category + str(fileIndex) + ".jpg"
                                                cv2.imwrite(imgSavePath, cornerImage)
                                                fileIndex += 1
                                

                            '''
                            cv2.drawContours(imgOri, squareContours8, -1, (0,255,0), 3)
                            cv2.imshow('window', imgOri)
                            cv2.waitKey(0)
                            '''

if __name__ == '__main__':
    main()




