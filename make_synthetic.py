import os
import cv2

import numpy as np

from tqdm import tqdm

backgrounds_dir = "/home/vid/hdd/file/project/240-EVRAZ/data/attrs/syntetic/backgrounds/"
templates_dir = "/home/vid/hdd/file/project/240-EVRAZ/data/attrs/syntetic/welding_helmet/"
output_dir = "/home/vid/hdd/file/project/240-EVRAZ/data/attrs/syntetic/result/"

angle_step = 90
angles_range = list(range(0, 360, angle_step))

image_count = 0

background_images = os.listdir(backgrounds_dir)
template_files = os.listdir(templates_dir)
template_images = sorted([x for x in template_files if ".png" in x])
template_labels = sorted([x for x in template_files if ".txt" in x])

for background in tqdm(background_images):

    background_image_path = "{}/{}".format(backgrounds_dir, background)
    background_src = cv2.imread(background_image_path, cv2.IMREAD_UNCHANGED)

    background_src_height, background_src_width = background_src.shape[:2]

    for template_img, template_txt in tqdm(zip(template_images, template_labels), leave=False):

        template_image_path = "{}/{}".format(templates_dir, template_img)
        template_label_path = "{}/{}".format(templates_dir, template_txt)

        template_image_src = cv2.imread(template_image_path, cv2.IMREAD_UNCHANGED)
        boxes = {}

        with open(template_label_path) as tlp:

            label_lines = tlp.readlines()

        for line in label_lines:

            class_id, rel_x_center, rel_y_center, rel_width, rel_height = line.strip().split(" ")

            if class_id not in boxes.keys():
                boxes[class_id] = []

            abs_x_center = int(float(rel_x_center) * template_image_src.shape[1])
            abs_y_center = int(float(rel_y_center) * template_image_src.shape[0])
            abs_width = int(float(rel_width) * template_image_src.shape[1])
            abs_height = int(float(rel_height) * template_image_src.shape[0])

            abs_xmin = int(abs_x_center - abs_width / 2)
            abs_ymin = int(abs_y_center - abs_height / 2)
            abs_xmax = abs_xmin + abs_width
            abs_ymax = abs_ymin + abs_height

            boxes[class_id].append(np.array([abs_xmin, abs_ymin, abs_xmax, abs_ymax]))

        for angle in tqdm(angles_range, leave=False):

            output_img_path = "{}{:06d}.jpg".format(output_dir, image_count)
            output_txt_path = "{}{:06d}.txt".format(output_dir, image_count)

            current_background = background_src.copy()

            imgHeight, imgWidth = template_image_src.shape[0], template_image_src.shape[1]
            center_y, center_x = imgHeight // 2, imgWidth // 2

            rotationMatrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)

            cosofRotationMatrix = np.abs(rotationMatrix[0, 0])
            sinofRotationMatrix = np.abs(rotationMatrix[0, 1])

            if int(cosofRotationMatrix) == 1 or int(sinofRotationMatrix) == 1:
                temp_ = cosofRotationMatrix
                cosofRotationMatrix = sinofRotationMatrix
                sinofRotationMatrix = temp_

            newImageHeight = int((imgHeight * sinofRotationMatrix) + (imgWidth * cosofRotationMatrix)) + 50
            newImageWidth = int((imgHeight * cosofRotationMatrix) + (imgWidth * sinofRotationMatrix)) + 50

            rotationMatrix[0, 2] += (newImageWidth / 2) - center_x
            rotationMatrix[1, 2] += (newImageHeight / 2) - center_y

            rotated_image = cv2.warpAffine(template_image_src, rotationMatrix, (newImageWidth, newImageHeight))
            rotated_image_height, rotated_image_width = rotated_image.shape[:2]

            if rotated_image_width >= background_src_width:
                reduction_factor_width = background_src_width / rotated_image_width
                rotated_image = cv2.resize(rotated_image, (int(rotated_image.shape[1] * reduction_factor_width) - 1,
                                                           int(rotated_image.shape[0] * reduction_factor_width) - 1))

            if rotated_image_height >= background_src_height:
                reduction_factor_height = background_src_height / rotated_image_height
                rotated_image = cv2.resize(rotated_image, (int(rotated_image.shape[1] * reduction_factor_height) - 1,
                                                           int(rotated_image.shape[0] * reduction_factor_height) - 1))

            rotated_image_height, rotated_image_width = rotated_image.shape[:2]

            pasted_xmin = np.random.randint(0, background_src_width - rotated_image_width)
            pasted_ymin = np.random.randint(0, background_src_height - rotated_image_height)

            y1, y2 = pasted_ymin, pasted_ymin + rotated_image_height
            x1, x2 = pasted_xmin, pasted_xmin + rotated_image_width

            alpha_s = rotated_image[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(0, 3):
                rotated_image[:, :, c] = (
                        alpha_l * current_background[y1:y2, x1:x2, c] + alpha_s * rotated_image[:, :, c])

            cv2.imwrite(output_img_path, rotated_image)

            for key in boxes.keys():

                boxes_to_process = boxes[key]

                for box in boxes_to_process:

                    points = np.array([[box[0], box[1]],
                                       [box[0], box[3]],
                                       [box[2], box[1]],
                                       [box[2], box[3]],
                                       ])

                    ones = np.ones(shape=(len(points), 1))
                    points_ones = np.hstack([points, ones])

                    transformed_points = rotationMatrix.dot(points_ones.T).T

                    box_points = []

                    for tp in transformed_points:
                        x = float(tp[0])
                        y = float(tp[1])
                        box_points.append([x, y])

                    bbox = cv2.boundingRect(np.array(box_points, dtype=np.int32))
                    bbox_list = [bbox[0], bbox[1], bbox[2], bbox[3]]

                    if angle % 90 != 0:
                        bbox_list[2] = int(bbox_list[2] * 0.9)
                        bbox_list[3] = int(bbox_list[3] * 0.9)

                    obj_x_center = bbox_list[0] + bbox_list[2] // 2
                    obj_y_center = bbox_list[1] + bbox_list[3] // 2

                    rel_obj_x_center = obj_x_center / rotated_image_width
                    rel_obj_y_center = obj_y_center / rotated_image_height
                    rel_obj_width = bbox_list[2] / rotated_image_width
                    rel_obj_height = bbox_list[3] / rotated_image_height

                    label_line = "{} {} {} {} {}\n".format(
                        key,
                        rel_obj_x_center,
                        rel_obj_y_center,
                        rel_obj_width,
                        rel_obj_height,
                    )

                    with open(output_txt_path, "a") as otp:

                        otp.write(label_line)

            image_count += 1
