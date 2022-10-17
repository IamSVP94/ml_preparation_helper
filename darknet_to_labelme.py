import os
import cv2
import json
import base64

from tqdm import tqdm

target_labelme_version = "5.0.1"

target_dirs = ["/home/vid/hdd/file/project/256_МешкиПодсчет/datasetv6mini", ]
class_names_path = "/home/vid/hdd/file/project/256_МешкиПодсчет/cfgv6/obj.names"

with open(class_names_path) as cnp:
    class_names = [x.strip() for x in cnp.readlines()]
for td in target_dirs:
    print(f"[INFO] Processing folder: {td}")
    out_dir = td + "_labelme"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    target_files = os.listdir(td)
    target_imgs = [x for x in target_files if ".jpg" in x]
    target_txts = [x for x in target_files if ".txt" in x]

    for img_name in tqdm(target_imgs):
        img_path = os.path.join(td, img_name)
        img_src = cv2.imread(img_path)
        img_src_height, img_src_width = img_src.shape[:2]

        out_json = {}
        out_json["version"] = target_labelme_version
        out_json["flags"] = {}
        out_json["shapes"] = []
        out_json["imagePath"] = img_name
        # out_json["imageData"] = base64.b64encode(img_src).decode()
        out_json["imageData"] = None
        out_json["imageHeight"] = img_src_height
        out_json["imageWidth"] = img_src_width

        txt_name = img_name.replace(".jpg", ".txt")

        if txt_name in target_txts:
            txt_path = os.path.join(td, txt_name)

            with open(txt_path) as tp:
                label_lines = [x.strip() for x in tp.readlines()]

            for ll in label_lines:
                class_id, rel_center_x, rel_center_y, rel_width, rel_height = ll.split(" ")

                abs_center_x = float(rel_center_x) * img_src_width
                abs_center_y = float(rel_center_y) * img_src_height
                abs_width = float(rel_width) * img_src_width
                abs_height = float(rel_height) * img_src_height

                xmin = abs_center_x - abs_width / 2
                ymin = abs_center_y - abs_height / 2
                xmax = xmin + abs_width
                ymax = ymin + abs_height

                if xmin < 0:
                    xmin = 0
                if ymin < 0:
                    ymin = 0

                if xmax > img_src_width:
                    xmax = img_src_width
                if ymax > img_src_height:
                    ymax = img_src_height
                obj_rect = {}
                obj_rect["label"] = class_names[int(class_id)]
                obj_rect["points"] = [[xmin, ymin], [xmax, ymax]]
                obj_rect["group_id"] = None
                obj_rect["shape_type"] = "rectangle"
                obj_rect["flags"] = {}
                out_json["shapes"].append(obj_rect)
        out_json_name = img_name.replace(".jpg", ".json")
        out_json_path = os.path.join(out_dir, out_json_name)
        with open(out_json_path, "w") as j:
            json.dump(out_json, j, ensure_ascii=False, indent=2)
        out_img_path = os.path.join(out_dir, img_name)
        cv2.imwrite(out_img_path, img_src)
