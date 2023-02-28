import argparse
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2


def render(x, y, t, p, shape):
    img = np.full(shape=shape + [3], fill_value=255, dtype="uint8")
    img[y, x, :] = 0
    img[y, x, p] = 255
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser("""Generate events from a high frequency video stream""")
    parser.add_argument("--input_dir", default="")
    parser.add_argument("--input_label_dir", default="")
    parser.add_argument("--shape", nargs=2, default=[480, 256]) # [480, 256],[256, 320]
    args = parser.parse_args()

    # event_files = sorted(glob.glob(os.path.join(args.input_dir, "outdoors_climbing_02*.npy")))
    # label_files = sorted(glob.glob(os.path.join(args.input_label_dir, "outdoors_climbing_02*.npy")))
    event_files = sorted(glob.glob(os.path.join(args.input_dir, "outdoors_slalom_00*.npy")))
    label_files = sorted(glob.glob(os.path.join(args.input_label_dir, "outdoors_slalom_00*.npy")))
    fig, ax = plt.subplots()
    events = np.load(event_files[200])
    label = np.load(label_files[200])
    u = label[0, :].astype(int)
    v = label[1, :].astype(int)
    mask = label[2, :]
    skeleton_parent_ids = [0, 0, 1, 2, 3, 1, 5, 6, 1, 8, 9, 1, 11, 12, 0, 0, 14, 15]
    color = (255, 0, 0)
    img = render(events[:, 0], events[:, 1], events[:, 2], events[:, 3], shape=args.shape)
    for points in range(0, 18):
        pos1 = (u[points], v[points])
        pos2 = (u[skeleton_parent_ids[points]], v[skeleton_parent_ids[points]])
        if mask[points]:
            cv2.circle(img, pos1, 3, color, -1)  # plot key-points
            # cv2.putText(img, str(points), pos1, cv2.FONT_HERSHEY_COMPLEX,0.5,color,1)
            if mask[skeleton_parent_ids[points]]:
                cv2.line(img, pos1, pos2, color, 2, 8)  # plot skeleton
    handle = plt.imshow(img)
    plt.show(block=False)
    plt.pause(0.002)

    # for f in event_files[1:]:
    #     events = np.load(f)
    #     img = render(shape=args.shape, **events)
    #     handle.set_data(img)
    #     plt.pause(0.002)



