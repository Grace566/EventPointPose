import os
import pickle as pkl
import numpy as np
import cv2

if __name__ == '__main__':

    Dataset_dir = "I:/Dataset/3DPW"
    raw_dir = Dataset_dir + "/3DPW_origin/imageFiles"

    # Classify the validation set in the original dataset into the train set
    f = open((Dataset_dir + "/test.txt"), "r")
    path_name = f.readlines()
    f.close()

    count = 0
    for path_list in path_name:
        path = raw_dir + '/' + path_list.strip()
        count += len(os.listdir(path))
    print(count)


