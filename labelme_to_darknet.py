import os
import json
import cv2
import shutil

from tqdm import tqdm

labelme_dataset = "/home/vid/hdd/file/project/256_МешкиПодсчет/razmetka10102022/labelme/"
darknet_dataset = "/home/vid/hdd/file/project/256_МешкиПодсчет/razmetka10102022/labelme2darknet/"

select_by = "image" # on of: "json" or "image"

class_id = {
    "bag": 0,
    # "welding helmet": 3,
    # "yellow vest": 5,
}

files_list = os.listdir(labelme_dataset)

if select_by == "json":
    
    json_list = [x for x in files_list if ".json" in x]

    for j in tqdm(json_list):

        watched_classes = []

        orig_image_path = "{}{}.jpg".format(labelme_dataset, j.split(".")[0])
        orig_image = cv2.imread(orig_image_path)
        orig_image_height, orig_image_width = orig_image.shape[:2]

        with open("{}{}".format(labelme_dataset, j)) as label_json:
            labels = json.load(label_json)["shapes"]

        if labels == []:

            darknet_label_path = os.path.join(darknet_dataset, j.split(".")[0] + ".txt")

            with open(darknet_label_path, "a") as dl:

                dl.write("")

            new_img_path = "{}{}.jpg".format(darknet_dataset, j.split(".")[0])
            shutil.copy(orig_image_path, new_img_path)

            continue

        for l in labels:

            class_name = l["label"]

            points = l["points"]

            abs_xmin = min(int(points[0][0]), int(points[1][0]))
            abs_ymin = min(int(points[0][1]), int(points[1][1]))
            abs_xmax = max(int(points[0][0]), int(points[1][0]))
            abs_ymax = max(int(points[0][1]), int(points[1][1]))

            abs_width = abs_xmax - abs_xmin
            abs_height = abs_ymax - abs_ymin
            abs_center_x = abs_width / 2 + abs_xmin
            abs_center_y = abs_height / 2 + abs_ymin

            rel_width = abs_width / orig_image_width
            rel_height = abs_height / orig_image_height
            rel_center_x = abs_center_x / orig_image_width
            rel_center_y = abs_center_y / orig_image_height

            class_idx = class_id[class_name]

            darknet_label_path = os.path.join(darknet_dataset, j.split(".")[0] + ".txt")

            with open(darknet_label_path, "a") as dl:

                dl.write("{} {} {} {} {}\n".format(
                                                    class_idx,
                                                    rel_center_x,
                                                    rel_center_y,
                                                    rel_width,
                                                    rel_height
                                                ))

        
        new_img_path = "{}{}.jpg".format(darknet_dataset, j.split(".")[0])
        shutil.copy(orig_image_path, new_img_path)

if select_by == "image":

    img_ext = ".jpg"
    
    img_list = [x for x in files_list if img_ext in x]
    json_list = [os.path.join(labelme_dataset, x) for x in files_list if ".json" in x]

    for i in tqdm(img_list):

        orig_image_path = "{}{}".format(labelme_dataset, i)
        orig_image = cv2.imread(orig_image_path)
        orig_image_height, orig_image_width = orig_image.shape[:2]

        json_path = "{}{}.json".format(labelme_dataset, i.split(".")[0])

        if json_path in json_list:

            with open(json_path) as label_json:
                labels = json.load(label_json)["shapes"]

            for l in labels:

                class_name = l["label"]

                points = l["points"]

                abs_xmin = min(int(points[0][0]), int(points[1][0]))
                abs_ymin = min(int(points[0][1]), int(points[1][1]))
                abs_xmax = max(int(points[0][0]), int(points[1][0]))
                abs_ymax = max(int(points[0][1]), int(points[1][1]))

                abs_width = abs_xmax - abs_xmin
                abs_height = abs_ymax - abs_ymin
                abs_center_x = abs_width / 2 + abs_xmin
                abs_center_y = abs_height / 2 + abs_ymin

                rel_width = abs_width / orig_image_width
                rel_height = abs_height / orig_image_height
                rel_center_x = abs_center_x / orig_image_width
                rel_center_y = abs_center_y / orig_image_height

                class_idx = class_id[class_name]

                darknet_label_path = os.path.join(darknet_dataset, i.split(".")[0] + ".txt")

                with open(darknet_label_path, "a") as dl:

                    dl.write("{} {} {} {} {}\n".format(
                                                        class_idx,
                                                        rel_center_x,
                                                        rel_center_y,
                                                        rel_width,
                                                        rel_height
                                                    ))
                        
        else:

            darknet_label_path = os.path.join(darknet_dataset, i.split(".")[0] + ".txt")

            with open(darknet_label_path, "a") as dl:

                dl.write("")
        
        new_img_path = "{}{}".format(darknet_dataset, i)
        shutil.copy(orig_image_path, new_img_path)
