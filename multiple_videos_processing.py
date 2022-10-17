import os
import sys
import cv2
import time

import numpy as np

from tqdm import tqdm

# --------------------------------------------------------------------------------------------------
# Darknet - Opencv initialization
# --------------------------------------------------------------------------------------------------
# config_path =   "/home/vid/hdd/file/project/RUS_AGRO_crops/piglet_dataset/cfg/yolov4-custom.cfg"
# names_path =    "/home/vid/hdd/file/project/RUS_AGRO_crops/piglet_dataset/cfg/obj.names"
# weight_path =   "/home/vid/hdd/file/project/RUS_AGRO_crops/piglet_dataset/backup/yolov4-custom_1000.weights"
config_path = "/home/vid/hdd/file/project/NLMK_SILA/cfgnew/yolov4-custom.cfg"
names_path = "/home/vid/hdd/file/project/NLMK_SILA/cfgnew/obj.names"
weight_path = "/home/vid/hdd/file/project/NLMK_SILA/backup/yolov4-custom_146900.weights"
threshold = 0.5
nms_threshold = 0.5

with open(names_path, "rt") as names:
    names = names.read().rstrip("\n").split("\n")

colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for i in names]

net = cv2.dnn_DetectionModel(config_path, weight_path)

if cv2.cuda.getCudaEnabledDeviceCount():
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

net.setInputSize(320, 320)
net.setInputScale(1.0 / 255)
net.setInputSwapRB(True)


# --------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------

def main():
    # videos_source = sys.argv[1]
    # list_of_videos = []
    videos_source = '/home/vid/hdd/file/project/NLMK_SILA/imgs/NLMK_SILA_v25_attrs/New folder/test/'
    list_of_videos = []

    if os.path.isfile(videos_source) and ".txt" in videos_source:
        list_of_videos = [x.strip() for x in open(videos_source).readlines()]

    if os.path.isdir(videos_source):
        list_of_videos = [os.path.join(videos_source, x) for x in os.listdir(videos_source)]

    assert list_of_videos != [], "[WARN] Videos source is empty or not exist"

    current_time = time.localtime()
    hour_ = current_time[3]
    min_ = current_time[4]
    sec_ = current_time[5]

    out_path = f"out_{hour_}{min_}{sec_}"

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    for video in tqdm(list_of_videos):

        cap = cv2.VideoCapture(video)

        video_name = video.split("/")[-1]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = "{}/{}".format(out_path, video_name)
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(out_path)

        while cap.isOpened():

            ret, frame = cap.read()

            if ret:

                # startT = time.time()
                classes, confidences, boxes = net.detect(frame, confThreshold=threshold,
                                                         nmsThreshold=nms_threshold)  # Class detection
                # detect_time = int(round((time.time() - startT) * 1000))

                # print("\nImage: {}, Time: {} ms".format(image_path, detect_time))

                if classes == ():

                    out.write(frame)
                    continue

                else:

                    for cls, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):

                        if float(confidence) > threshold:

                            current_class = names[cls]
                            current_thresh = float(confidence) * 100

                            # print("Probability: {:.3f}, Class: {}".format(current_thresh, current_class))

                            fontType = cv2.FONT_HERSHEY_SIMPLEX
                            fontScale = 0.8

                            text = "{} ({:.3f}%)".format(current_class, current_thresh)

                            xmin = int(box[0])
                            ymin = int(box[1])
                            xmax = int(xmin + box[2])
                            ymax = int(ymin + box[3])

                            # bbox_height = int(box[3])
                            # bbox_width = int(box[2])
                            # area = bbox_height * bbox_width

                            # if area > 497250 or area < 100:

                            #     continue

                            if xmin < 0:
                                xmin = 0

                            if xmax > frame.shape[1]:
                                xmax = frame.shape[1]

                            if ymin < 0:
                                ymin = 0

                            if ymax > frame.shape[0]:
                                ymax = frame.shape[0]

                            cv2.rectangle(
                                frame,
                                (xmin, ymin),
                                (xmax, ymax),
                                colors[cls],
                                2
                            )

                            cv2.putText(
                                frame,
                                text,
                                (xmin, ymin - 10),
                                fontType,
                                fontScale,
                                (0, 0, 0),
                                2
                            )

                    out.write(frame)

            else:

                break

        cap.release()
        out.release()

    print("Done!")


if __name__ == "__main__":
    main()
