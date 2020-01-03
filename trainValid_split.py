import os
import numpy as np
import shutil

# # Creating Train / Val / Test folders (One time use)
ori_dir = r"E:\Estek-AIProject\INTEL-SOOP\RaspberryPiFolder\Images\Training\TrainingSet31_12_19\CornersExtracted64px_augmented_combined"
dest_dir = r"E:\Estek-AIProject\INTEL-SOOP\RaspberryPiFolder\Images\Training\TrainingSet31_12_19\trainValidSplit"

categories = []
splitTypes = ['train','valid']
trainingRatio = 0.8

#get number of categories/labels in directory
for filename in os.listdir(ori_dir):
    categories.append(filename)

#get number of splits
for splitType in splitTypes:
    os.makedirs(dest_dir + "\\" + splitType)
    for category in categories:
        os.makedirs(dest_dir + "\\" + splitType + "\\" + category)

#for each category/label, perform the split
for category in categories:
    allFileNames = os.listdir(ori_dir + "\\" + category)
    np.random.shuffle(allFileNames)
    train_FileNames, val_FileNames= np.split(np.array(allFileNames),
                                                            [int(len(allFileNames)*trainingRatio)])


    train_FileNames = [ori_dir +'\\' + category + "\\" + name for name in train_FileNames.tolist()]
    val_FileNames = [ori_dir +'\\' + category + "\\" + name for name in val_FileNames.tolist()]

    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))

    # Copy-pasting images
    counter = 0
    for name in train_FileNames:
        dest = dest_dir + "\\" + "train" + "\\" + category + "\\" + category + str(counter) + ".jpg"
        shutil.copy(name, dest)
        counter += 1

    counter = 0
    for name in val_FileNames:
        dest = dest_dir + "\\" + "valid" + "\\" + category + "\\" + category + str(counter) + ".jpg"
        shutil.copy(name, dest)
        counter += 1
