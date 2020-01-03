import cv2
import numpy as np

def adjustBrightness(inputImg, jitterAlpha, jitterBeta):
    listOfImages = []
    imgAug1 = cv2.convertScaleAbs(img, alpha=1 + jitterAlpha, beta= jitterBeta)
    imgAug2 = cv2.convertScaleAbs(img, alpha= 1 - jitterAlpha, beta = jitterBeta)
    listOfImages.append(imgAug1)
    listOfImages.append(imgAug2)
    return listOfImages

def addBlur(inputListOfImages, blur):
    listOfImages = []

    for image in inputListOfImages:
        imgAug1 = cv2.GaussianBlur(image, (5,5) )
        listOfImages.append(imgAug1)

    return listOfImages

IMG_PATH = r"E:\Estek-AIProject\INTEL-SOOP\RaspberryPiFolder\Images\Training\Extracted\XCC\SOOP\SOOP100.jpg"
img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)

cv2.namedWindow("window")
cv2.imshow("window", img)
print(img)
cv2.waitKey()

imgAug = cv2.convertScaleAbs(img, alpha=1.1, beta=2)
cv2.imshow("window", imgAug)
cv2.waitKey()

imgAug = cv2.convertScaleAbs(img, alpha=0.9, beta=2)
cv2.imshow("window", imgAug)
cv2.waitKey()

imgAug = cv2.convertScaleAbs(img, alpha=0.8, beta=2)
cv2.imshow("window", imgAug)
cv2.waitKey()

imgAug = cv2.convertScaleAbs(img, alpha=1.2, beta=2)
cv2.imshow("window", imgAug)
cv2.waitKey()



