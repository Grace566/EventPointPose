import os
import shutil
import pickle as pkl
import numpy as np
import cv2


def merge_event(event_stamp, img_stamp, i, events_path):    # Merge event according to the current timestamp
    start_stamp = img_stamp[i]
    end_stamp = img_stamp[i + 2]
    indexes = np.array(np.where((event_stamp >= start_stamp) * (event_stamp <= end_stamp))).reshape(-1)
    event = np.load((events_path + '/' + str(indexes[0]).zfill(10) + '.npz'), allow_pickle=True)
    event_x = event[event.files[0]]
    event_y = event[event.files[1]]
    event_t = event[event.files[2]]
    event_p = event[event.files[3]]
    count = 0
    for index in indexes:
        if count > 0:
            event_index = np.load((events_path + '/' + str(index).zfill(10) + '.npz'), allow_pickle=True)
            event_x = np.append(event_x, event_index[event_index.files[0]])
            event_y = np.append(event_y, event_index[event_index.files[1]])
            event_t = np.append(event_t, event_index[event_index.files[2]])
            event_p = np.append(event_p, event_index[event_index.files[3]])
        count += 1
    return event_x, event_y, event_t, event_p


if __name__ == '__main__':

    Dataset_dir = "I:/Dataset/3DPW"
    raw_dir = Dataset_dir + "/3DPW_origin/sequenceFiles"
    img_dir = Dataset_dir + "/3DPW_EPC/imageFiles"
    events_dir = Dataset_dir + "/3DPW_EPC/events"
    label_dir = Dataset_dir + "/3DPW_EPC/label"
    timestamp_dir = Dataset_dir + "/3DPW_EPC/imageFiles_Upsample"
    output_dir = Dataset_dir + "/Event_3DPW"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # temp = np.load("I:/Dataset/EventPointPose_Dataset/train/data/S1_session1_mov1_frame0_cam0.npy")
    # temp = np.load("I:/Dataset/EventPointPose_Dataset/train/label/S1_session1_mov1_frame0_cam0_label.npy")
    # temp = np.load("I:/Dataset/3DPW/Event_3DPW/train/data/courtyard_backpack_00_000001.npy")
    # temp = np.load("I:/Dataset/3DPW/Event_3DPW/train/label/courtyard_backpack_00_000001.npy")
    # Classify the validation set in the original dataset into the train set
    f = open((Dataset_dir + "/single_player.txt"), "r")
    path_name = f.readlines()
    f.close()

    # Obtain train and test lists
    train_list = []
    test_list = []
    for split_list in os.listdir(raw_dir):
        split_dir = raw_dir + '/' + split_list + '/'
        if (split_list == "train") or (split_list == "validation"):
            for file in os.listdir(split_dir):
                if (file.split(".")[0] + '\n') in path_name:
                    train_list.append(file.split(".")[0])
        elif split_list == "test":
            for file in os.listdir(split_dir):
                if (file.split(".")[0] + '\n') in path_name:
                    test_list.append(file.split(".")[0])

    # # Data/Events
    # fps = 30
    # for train in train_list:    # Each sequence
    #     events_path = events_dir + '/' + train
    #     output_path = output_dir + '/train/data'
    #     if not os.path.exists(output_path):
    #         os.makedirs(output_path)
    #
    #     img_len = len(os.listdir((img_dir + '/' + train + '/imgs')))
    #     img_stamp = np.arange(0, img_len * (1 / fps), 1 / fps)
    #     f = open((timestamp_dir + '/' + train + '/timestamps.txt'), "r")
    #     text = f.readlines()
    #     event_stamp = np.array([line.strip("\n") for line in text], dtype=np.float)[0:-1]
    #     f.close()
    #     for i in range(img_len - 2):        # Timestamps for each label in the sequence
    #         event_x, event_y, event_t, event_p = merge_event(event_stamp, img_stamp, i, events_path)
    #         event = np.concatenate((event_x.reshape(-1, 1), event_y.reshape(-1, 1), event_t.reshape(-1, 1),
    #                                 event_p.reshape(-1, 1)), axis=1)
    #         np.save((output_path + '/' + train + '_' + str(i+1).zfill(6)+'.npy'), event)
    #
    # for test in test_list:
    #     events_path = events_dir + '/' + test
    #     output_path = output_dir + '/test/data'
    #     if not os.path.exists(output_path):
    #         os.makedirs(output_path)
    #
    #     img_len = len(os.listdir((img_dir + '/' + test + '/imgs')))
    #     img_stamp = np.arange(0, img_len * (1 / fps), 1 / fps)
    #     f = open((timestamp_dir + '/' + test + '/timestamps.txt'), "r")
    #     text = f.readlines()
    #     event_stamp = np.array([line.strip("\n") for line in text], dtype=np.float)[0:-1]
    #     f.close()
    #     for i in range(img_len - 2):  # Timestamps for each label in the sequence
    #         event_x, event_y, event_t, event_p = merge_event(event_stamp, img_stamp, i, events_path)
    #         event = np.concatenate((event_x.reshape(-1, 1), event_y.reshape(-1, 1), event_t.reshape(-1, 1),
    #                                 event_p.reshape(-1, 1)), axis=1)
    #         np.save((output_path + '/' + test + '_' + str(i + 1).zfill(6) + '.npy'), event)

    # Label
    for train in train_list:    # Each sequence
        label_path = label_dir + '/' + train + '.npy'
        seq_label = np.load(label_path)
        output_path = output_dir + '/train/label'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for i in range(seq_label.shape[0] - 2):
            label = seq_label[i+1, :, :]
            np.save((output_path + '/' + train + '_' + str(i + 1).zfill(6) + '.npy'), label)

    for test in test_list:
        label_path = label_dir + '/' + test + '.npy'
        seq_label = np.load(label_path)
        output_path = output_dir + '/test/label'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for i in range(seq_label.shape[0] - 2):
            label = seq_label[i+1, :, :]
            np.save((output_path + '/' + test + '_' + str(i + 1).zfill(6) + '.npy'), label)