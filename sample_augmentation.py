# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 16:43:36 2018

@author: Ru Sern
"""


# Python3 code to rename and augment multiple image files in a directory or folder 

  
# importing os module 
import os 
import shutil
import cv2
from matplotlib import pyplot as plt
import numpy as np


def nameNumbering(Count):
    name = str(Count)
    str0 = '0'
    if len(name) < 6:
        num = 6 - len(name)
        for x in range(num - 1):
            str0 = str0 + '0'
            name = str0 + name
    return name

def adjustBrightness(inputImgList, jitterAlpha, jitterBeta):
    listOfImages = []
    for image in inputImgList:
        imgAug1 = cv2.convertScaleAbs(image, alpha=1 + jitterAlpha, beta= jitterBeta)
        imgAug2 = cv2.convertScaleAbs(image, alpha= 1 - jitterAlpha, beta = jitterBeta)
        listOfImages.append(imgAug1)
        listOfImages.append(imgAug2)
    return listOfImages

def addBlur(inputListOfImages, kernelSize):
    listOfImages = []
    for image in inputListOfImages:
        imgAug1 = cv2.blur(image, (kernelSize,kernelSize))
        listOfImages.append(imgAug1)

    return listOfImages

def augmentBrightnessAndBlur(inputImgList, jitterAlpha, jitterBeta, kernelSize):
    listOfImages = []
    for image in inputImgList:
        image = [image] #add image to a list, functions take in lists as input
        listOfImagesChangedBrightness = adjustBrightness(image, jitterAlpha, jitterBeta)
        listOfImagesChangedBrightness2 = adjustBrightness(image,jitterAlpha*2, jitterBeta*2)

        listOfImagesChangedBrightness.extend(listOfImagesChangedBrightness2)
        listOfImagesToBlur = listOfImagesChangedBrightness
        listOfBlurredImages = addBlur(listOfImagesToBlur, kernelSize)
        listOfImages.extend(listOfBlurredImages)
    
    return listOfImages





# Function to rename and augment multiple files 
def main():
    dir_path = r'E:\Estek-AIProject\INTEL-SOOP\RaspberryPiFolder\Images\Training\TrainingSet31_12_19\CornersExtracted64px'
    #main directory level
    dest_dir_path = r'E:\Estek-AIProject\INTEL-SOOP\RaspberryPiFolder\Images\Training\TrainingSet02_01_20\ExtractedCorners64px_Augmented'
    fileIndex = 0
    for filename in os.listdir(dir_path):
        #unit type level (XCC,BHP,LCC)
        file_path = os.path.join(dir_path, filename)

        if os.path.isdir(file_path):
            destFilePath = os.path.join(dest_dir_path, filename)
            for filename in os.listdir(file_path):
                #SOOP/GOOD level
                
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
                            img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
                            img_r = cv2.resize(img, (128,128))

                            rows,cols, channels = img_r.shape
                            img90 = np.rot90(m = img_r, k = 1)
                            img180 = np.rot90(m = img_r, k = 2)
                            img270 = np.rot90(m = img_r, k = 3)
                        
                            img_flip = cv2.flip(img_r,1)
                            img90f = np.rot90(m = img_flip, k = 1)
                            img180f = np.rot90(m = img_flip, k = 2)
                            img270f = np.rot90(m = img_flip, k = 3)

                            img_aug = [img_r, img90, img180, img270, img_flip,img90f, img180f, img270f]
                            for i in img_aug:
                                j = [i]
                                listOfAugmentedImages = augmentBrightnessAndBlur(j, 0.1, 2, 5)      
                                listOfAugmentedImages.append(i)                         

                                for augmentedImage in listOfAugmentedImages:
                                    index = nameNumbering(fileIndex)
                                    save_path = targetSubDirPath + '\\' + category + index + ".jpg"
                                    cv2.imwrite(save_path,augmentedImage)
                                    fileIndex +=1
                        cv2.waitKey(0)

if __name__ == '__main__': 
    main()
