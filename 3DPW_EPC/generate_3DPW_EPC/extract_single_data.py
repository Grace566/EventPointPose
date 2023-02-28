import os
import pickle as pkl
import numpy as np
import cv2


if __name__ == '__main__':

    Dataset_dir = "I:/Dataset/3DPW"
    input_dir = Dataset_dir + "/3DPW_origin/imageFiles"
    label_dir = Dataset_dir + "/3DPW_origin/sequenceFiles"
    output_dir = Dataset_dir + "/3DPW_EPC/imageFiles"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract names of sequence with a single person
    path_name = []
    for root, dirs, files in os.walk(label_dir):
        for file in files:
            path = os.path.join(root, file)
            seq = pkl.load(open(path, 'rb'), encoding='bytes')
            if seq[b'poses2d'].__len__() == 1:
                path_name.append(file.split(".")[0])

    f = open((Dataset_dir + "/single_player.txt"), "w")
    for line in path_name:
        f.write(line + '\n')
    f.close()

    # Extract and reshape the single-player dataset according to the format of rpg_vid
    # Meanwhile, resize the images
    for Seq_list in os.listdir(input_dir):
            if Seq_list in path_name:
                Seqinpath = input_dir + '/' + Seq_list + '/'
                Seqoutpath = output_dir + '/' + Seq_list + '/imgs/'
                assert not os.path.exists(Seqoutpath), 'The output directory must not exist'

                os.makedirs(Seqoutpath)
                for img_list in os.listdir(Seqinpath):
                    img = cv2.imread(Seqinpath + img_list)
                    height, width = img.shape[0], img.shape[1]
                    if (height == 1080) and (width == 1920):
                        height_resize = 480
                        width_resize = 270
                    elif (height == 1920) and (width == 1080):
                        height_resize = 270
                        width_resize = 480
                    # resize
                    img_resize = cv2.resize(img, (height_resize, width_resize), interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite((Seqoutpath + "/img_" + img_list[6:-4].zfill(8) + ".png"), img_resize)
                # copy_dir(Seqinpath, Seqoutpath)
                file = open(output_dir + '/' + Seq_list +'/fps.txt', 'w')
                file.write("30")
                file.close()


    # seq_name = 'courtyard_backpack_00'
    # datasetDir = 'I:/Dataset/3DPW'
    # file = os.path.join(datasetDir,'sequenceFiles/train',seq_name+'.pkl')
    # seq = pkl.load(open(file,'rb'))

