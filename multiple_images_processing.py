import os
import sys
import cv2
import time
import numpy as np

from tqdm import tqdm

# --------------------------------------------------------------------------------------------------
# Darknet - Opencv initialization
# --------------------------------------------------------------------------------------------------
config_path = "/home/vid/hdd/file/project/RUS_AGRO_crops/piglet_dataset/cfg/yolov4-custom.cfg"
names_path = "/home/vid/hdd/file/project/RUS_AGRO_crops/piglet_dataset/cfg/obj.names"
weight_path = "/home/vid/hdd/file/project/RUS_AGRO_crops/piglet_dataset/backup/yolov4-custom_last.weights"

threshold = 0.00001
nms_threshold = 0.5

with open(names_path, "rt") as an:
    names = an.read().rstrip("\n").split("\n")

colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for i in names]

net = cv2.dnn_DetectionModel(config_path, weight_path)

if cv2.cuda.getCudaEnabledDeviceCount():
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

net.setInputSize(320, 320)
net.setInputScale(1.0 / 255)
net.setInputSwapRB(True)


# --------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------

def main():
    images_source = sys.argv[1]
    list_of_images = []

    if os.path.isfile(images_source) and ".txt" in images_source:
        list_of_images = [x.strip() for x in open(images_source).readlines()]

    if os.path.isdir(images_source):
        list_of_images = [os.path.join(images_source, x) for x in os.listdir(images_source) if ".jpg" in x]

    assert list_of_images != [], "[WARN] Images source is empty or not exist"

    current_time = time.localtime()
    hour_ = current_time[3]
    min_ = current_time[4]
    sec_ = current_time[5]
    out_path = f"out_{hour_}{min_}{sec_}"

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    for img in tqdm(list_of_images):

        img_path = "{}".format(img)
        img_src = cv2.imread(img_path)

        classes, confidences, boxes = net.detect(img_src, confThreshold=threshold,
                                                 nmsThreshold=nms_threshold)  # Class detection

        if classes == ():
            out_img_path = "{}/{}".format(out_path, img.split("/")[-1])
            cv2.imwrite(out_img_path, img_src)
            continue

        for cls, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):

            if float(confidence) > threshold:

                current_class = names[cls]
                current_thresh = float(confidence) * 100

                fontType = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.5

                text = "{} ({:.3f}%)".format(current_class, current_thresh)

                xmin = int(box[0]) - 20
                ymin = int(box[1]) - 20
                xmax = int(xmin + box[2]) + 40
                ymax = int(ymin + box[3]) + 40

                if xmin < 0:
                    xmin = 0

                if xmax > img_src.shape[1]:
                    xmax = img_src.shape[1]

                if ymin < 0:
                    ymin = 0

                if ymax > img_src.shape[0]:
                    ymax = img_src.shape[0]

                cv2.rectangle(
                    img_src,
                    (xmin, ymin),
                    (xmax, ymax),
                    colors[cls],
                    2
                )

                cv2.putText(
                    img_src,
                    text,
                    (xmin, ymin - 10),
                    fontType,
                    fontScale,
                    (0, 0, 255),
                    1
                )

        out_img_path = "{}/{}".format(out_path, img.split("/")[-1])
        cv2.imwrite(out_img_path, img_src)

    print("Done!")


if __name__ == "__main__":
    main()
