import os
import cv2
from tqdm import tqdm
from pathlib import Path

original_yolo_dir = "/home/vid/hdd/file/project/RUS_AGRO_crops/piglet_dataset/imgs/"
img_paths = list(Path(original_yolo_dir).glob('*.jpg'))

for img_path in tqdm(img_paths):
    txt_path = img_path.with_suffix('.txt')
    with open(txt_path) as txt_label:
        label_lines = txt_label.readlines()
    if len(label_lines) == 0:
        continue  # pass empty

    img_src = cv2.imread(str(img_path))
    img_rotated_90 = cv2.rotate(img_src, cv2.ROTATE_90_CLOCKWISE)
    img_rotated_180 = cv2.rotate(img_src, cv2.ROTATE_180)

    output_lines_90 = []
    output_lines_180 = []

    for line in label_lines:
        class_id, rel_center_x, rel_center_y, rel_width, rel_height = line.strip().split(" ")

        output_line_90 = f"{class_id} {1 - float(rel_center_y)} {rel_center_x} {rel_height} {rel_width}\n"
        output_line_180 = f"{class_id} {1 - float(rel_center_x)} {1 - float(rel_center_y)} {rel_width} {rel_height}\n"
        output_lines_90.append(output_line_90)
        output_lines_180.append(output_line_180)

    out_img_path_90 = Path(original_yolo_dir) / f'{img_path.stem}__90.jpg'
    out_txt_path_90 = out_img_path_90.with_suffix('.txt')

    out_img_path_180 = Path(original_yolo_dir) / f'{img_path.stem}__180.jpg'
    out_txt_path_180 = out_img_path_180.with_suffix('.txt')

    cv2.imwrite(str(out_img_path_90), img_rotated_90)
    with open(out_txt_path_90, "a") as out_label_90:
        for line_90 in output_lines_90:
            out_label_90.write(line_90)

    cv2.imwrite(str(out_img_path_180), img_rotated_180)
    with open(out_txt_path_180, "a") as out_label_180:
        for line_180 in output_lines_180:
            out_label_180.write(line_180)
