import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

IMG_DIR = '/home/vid/hdd/file/project/256_МешкиПодсчет/datasetv4/'
iou_thresh = 0.3
NEW_IMG_DIR = Path(f'/home/vid/ssd/240-ЕВРАЗ/data/img_bad_iou={iou_thresh}/')
bad_imgs_path = f'/home/vid/ssd/240-ЕВРАЗ/data/cfg_pseudo/bad_files_{iou_thresh}.txt'
txts = list(Path(IMG_DIR).glob('*.txt'))
print(len(txts))


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1[1] < bb1[3]
    assert bb1[2] < bb1[4]
    assert bb2[1] < bb2[3]
    assert bb2[2] < bb2[4]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[1], bb2[1])
    y_top = max(bb1[2], bb2[2])
    x_right = min(bb1[3], bb2[3])
    y_bottom = min(bb1[4], bb2[4])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[3] - bb1[1]) * (bb1[4] - bb1[2])
    bb2_area = (bb2[3] - bb2[1]) * (bb2[4] - bb2[2])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def yolo2bbox(yolo_format, image_w, image_h, show_img=None):
    cl, x, y, w, h = map(float, yolo_format.split(' '))
    xmin = int((x - w / 2) * image_h)
    ymin = int((y - h / 2) * image_w)
    xmax = int(xmin + w * image_h)
    ymax = int(ymin + h * image_w)
    bbox = [int(cl), xmin, ymin, xmax, ymax]
    if show_img is not None:
        img = show_img.copy()
        cv2.rectangle(show_img, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
        plt.imshow(img)
        plt.show()
    return bbox


NEW_IMG_DIR.mkdir(parents=True, exist_ok=True)
bad_files = []
pbar = tqdm(txts)
for txt_path in pbar:
    with open(txt_path, 'r') as f:
        txts = [t.rstrip() for t in f.readlines()]
        img_path = txt_path.with_suffix('.jpg')
    if len(txts) > 1 and img_path.exists():
        img = cv2.imread(str(img_path))
        bboxes = [yolo2bbox(line, img.shape[0], img.shape[1]) for line in txts]
        for box1_idx, box1 in enumerate(bboxes):
            for box2_idx, box2 in enumerate(bboxes):
                if box1_idx == box2_idx:
                    continue
                iou = get_iou(box1, box2)
                if iou >= iou_thresh:
                    bad_txt_path = NEW_IMG_DIR / txt_path.name
                    bad_img_path = bad_txt_path.with_suffix('.jpg')
                    if txt_path.exists():
                        # shutil.move(txt_path, bad_txt_path)
                        # shutil.move(img_path, bad_img_path)
                        bad_files.append(bad_txt_path)
print(len(set(bad_files)))
# with open(bad_imgs_path, 'w') as f:
#     for file in set(bad_files):
#         f.write(f'{file}\n')
