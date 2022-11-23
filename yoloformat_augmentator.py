from pathlib import Path
import copy
import albumentations as A
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm

original_yolo_dir = "/home/vid/hdd/file/project/258-ЕврохимПломба/data/crop_imgs/data/full/"
new_yolo_dir = "/home/vid/hdd/file/project/258-ЕврохимПломба/data/crop_imgs/data/new_augmentations/"
img_paths = list(Path(original_yolo_dir).glob('*.png'))


def yolo2bbox(yolo_format, image_w, image_h, show_img=None):
    # cl, x, y, w, h = map(float, yolo_format.split(' '))
    cl, x, y, w, h = yolo_format
    xmin = int((x - w / 2) * image_h)
    ymin = int((y - h / 2) * image_w)
    xmax = int(xmin + w * image_h)
    ymax = int(ymin + h * image_w)
    bbox = [str(int(cl)), xmin, ymin, xmax, ymax]
    if show_img is not None:
        img = show_img.copy()
        cv2.rectangle(show_img, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
        plt.imshow(img)
        plt.show()
    return bbox


def draw_rect_yolo(img, bboxes, labels, show=False):
    image = img.copy()
    labels_ = copy.deepcopy(labels)
    bboxes_ = list(copy.deepcopy(bboxes))
    w, h, _ = image.shape
    for bbox, cl in zip(bboxes_, labels_):
        bbox = list(bbox)
        bbox.insert(0, str(int(float(cl))))
        cl, xmin, ymin, xmax, ymax = yolo2bbox(bbox, w, h)
        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), colors[int(cl)], 1)
    if show:
        plt.imshow(image)
        plt.show()
    return image


colors = [
    (238, 49, 106), (44, 4, 48), (123, 56, 128), (62, 240, 9), (52, 7, 44),
    (129, 37, 71), (46, 219, 227), (174, 200, 149), (26, 154, 146),
    (11, 117, 36), (15, 45, 179), (195, 222, 17), (167, 153, 192)
]

# Declare an augmentation pipeline
transform = A.Compose([
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=7, p=1),
    A.PiecewiseAffine(scale=(0.01, 0.015), nb_rows=1, nb_cols=20, interpolation=4, p=0.66),
    A.GridDistortion(num_steps=3, distort_limit=0.3, interpolation=1, border_mode=1, p=1),
    # ], bbox_params=A.BboxParams(format='coco')
], bbox_params=A.BboxParams(format='yolo', label_fields=['labels'])
)

new_yolo_dir = Path(new_yolo_dir)

for img_path in tqdm(img_paths):
    img = cv2.imread(str(img_path))
    w, h, _ = img.shape

    txt_path = img_path.with_suffix('.txt')
    bboxes, labels = [], []
    with open(str(txt_path), 'r') as rf:
        lines = rf.readlines()
        for line_idx, line in enumerate(lines):
            cl, x, y, w, h = map(float, line.rstrip().split(' '))
            bbox = [x, y, w, h]
            bboxes.append(bbox)
            labels.append(str(int(float(cl))))
    # draw_rect_yolo(img, bboxes, labels)
    transformed = transform(image=img, bboxes=bboxes, labels=labels)
    image = transformed['image']
    bboxes = transformed['bboxes']
    labels = transformed['labels']
    # draw_rect_yolo(image, bboxes, labels)

    new_img_path = new_yolo_dir / img_path.name
    new_txt_path = new_img_path.with_suffix('.txt')
    new_img_path.parent.mkdir(parents=True, exist_ok=True)
    with open(new_txt_path, 'w') as wf:
        for cl, box in zip(labels, bboxes):
            x, y, w, h = box
            wf.write(f'{cl} {x} {y} {w} {h}\n')
        cv2.imwrite(str(new_img_path), image)
