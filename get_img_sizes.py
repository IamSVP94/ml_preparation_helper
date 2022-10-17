from pathlib import Path
from tqdm import tqdm
import shutil
import cv2

IMG_DIR = '/home/vid/hdd/projects/PycharmProjects/keypoint_mmpose/temp/video/output_pro_affine_0510_2/body_topdown_heatmap_batch_1_mobilenetv2_coco_256x192/'
imgs = list(Path(IMG_DIR).glob('*.jpg'))
percent = (10, 10)  # h,w


def get_percentil_size(imgs, percent):
    heights, widths = [], []
    for img_idx, img_path in enumerate(tqdm(imgs)):
        img = cv2.imread(str(img_path))  # h,w,c
        heights.append(img.shape[0])
        widths.append(img.shape[1])
    heights = sorted(heights)
    widths = sorted(widths)

    part_h = int((len(heights) / 100 * percent[0]))
    part_w = int((len(heights) / 100 * percent[1]))

    min_percent_height = heights[part_h + 1]
    min_percent_width = widths[part_w + 1]
    return (part_h, part_w), min_percent_height, min_percent_width


bad_parts_hw, min_percent_height, min_percent_width = get_percentil_size(imgs, percent)

print(bad_parts_hw)
print(min_percent_height)
print(min_percent_width)

copy_counter = 0
NEW_IMG_DIR = Path(
    f'/home/vid/hdd/projects/PycharmProjects/keypoint_mmpose/temp/video/output_pro_affine_0510_2//data_minsize=({min_percent_height},{min_percent_width})/')
for img_idx, img_path in enumerate(tqdm(imgs)):
    img = cv2.imread(str(img_path))
    if img.shape[0] >= min_percent_height and img.shape[1] >= min_percent_width:
        parts = img_path.parts[len(Path(IMG_DIR).parts):]
        parts = '/'.join(parts)
        new_img_path = Path(NEW_IMG_DIR, parts)

        txt_path = img_path.with_suffix('.txt')
        new_txt_path = new_img_path.with_suffix('.txt')

        if img_path.exists() and txt_path.exists():
            new_img_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(img_path, new_img_path)
            shutil.copy(txt_path, new_txt_path)
            copy_counter += 1

print(f'original dataset size: {len(imgs)}')
print(f'new dataset size: {copy_counter}')
