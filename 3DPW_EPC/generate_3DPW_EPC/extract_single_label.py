import os
import pickle as pkl
import numpy as np
import cv2


if __name__ == '__main__':

    Dataset_dir = "I:/Dataset/3DPW"
    label_dir = Dataset_dir + "/3DPW_origin/sequenceFiles"
    label_output_dir = Dataset_dir + "/3DPW_EPC/label"
    upsampled_dir = Dataset_dir + "/3DPW_EPC/imageFiles_Upsample"
    if not os.path.exists(label_output_dir):
        os.makedirs(label_output_dir)

    # Extract pose2D from label(sequenceFiles) and resize it according to the size of upsampled image
    f = open((Dataset_dir + "/single_player.txt"), "r")
    path_name = f.readlines()
    f.close()
    for root, dirs, files in os.walk(label_dir):
        for file in files:
            if (file.split(".")[0] + '\n') in path_name:
                path = os.path.join(root, file)
                seq = pkl.load(open(path, 'rb'), encoding='bytes')
                label = seq[b'poses2d'][0]
                img_path = upsampled_dir + '/' + file.split(".")[0] + '/imgs/00000000.png'
                img = cv2.imread(img_path)
                height, width = img.shape[0], img.shape[1]
                cropsize = 7
                if (height == 480) and (width == 256):
                    label[:, 0, :] = np.maximum(np.round(label[:, 0, :] / 4) - cropsize, 0)
                    label[:, 1, :] = np.round(label[:, 1, :] / 4)
                elif (height == 256) and (width == 480):
                    label[:, 0, :] = np.round(label[:, 0, :] / 4)
                    label[:, 1, :] = np.maximum(np.round(label[:, 1, :] / 4) - cropsize, 0)

                # np.save((label_output_dir + "/" + file.split(".")[0] + ".npy"), label)