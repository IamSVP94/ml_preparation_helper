from pathlib import Path

import cv2
from tqdm import tqdm

IMG_LIST = '/home/vid/hdd/file/project/258-ЕврохимПломба/cfg3110/full.txt'

with open(IMG_LIST, 'r') as rf:
    files = list(map(lambda line: Path(line.strip()), rf.readlines()))

''' # crops
CFG_DIR = Path('/home/vid/hdd/file/project/258-ЕврохимПломба/models/crops/R0.1/')
CONFIG_PATH = CFG_DIR / 'crops_R0.1_31102022.cfg'
DATA_PATH = CFG_DIR / 'crops_R0.1_31102022.data'
NAMES_PATH = CFG_DIR / 'crops_R0.1_31102022.names'
WEIGHT_PATH = CFG_DIR / 'crops_R0.1_31102022.weights'
WINDOW_SIZE = (320, 320)
CONFIDENCE_THRESHOLD = .5
NMS_THRESHOLD = .5
# '''

# ''' # fulls
CFG_DIR = Path('/home/vid/hdd/file/project/258-ЕврохимПломба/models/1stage/R0/')
CONFIG_PATH = CFG_DIR / '1stage_27102022.cfg'
DATA_PATH = CFG_DIR / '1stage_27102022.data'
NAMES_PATH = CFG_DIR / '1stage_27102022.names'
WEIGHT_PATH = CFG_DIR / '1stage_27102022.weights'
WINDOW_SIZE = (416, 416)
CONFIDENCE_THRESHOLD = .5
NMS_THRESHOLD = .5
# '''

NET = cv2.dnn_DetectionModel(str(CONFIG_PATH), str(WEIGHT_PATH))

NET.setInputSize(WINDOW_SIZE)
NET.setInputScale(1.0 / 255)
NET.setInputSwapRB(True)
if cv2.cuda.getCudaEnabledDeviceCount():
    NET.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    NET.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

with open(NAMES_PATH, "rt") as names:
    NAMES = names.read().rstrip("\n").split("\n")
COLORS = [(238, 49, 106), (44, 4, 48), (123, 56, 128), (62, 240, 9),
          (52, 7, 44), (129, 37, 71), (46, 219, 227), (174, 200, 149), (26, 154, 146),
          (11, 117, 36), (15, 45, 179), (195, 222, 17), (167, 153, 192)]
for img_path in tqdm(files):
    img = cv2.imread(str(img_path))
    classes, confidences, boxes = NET.detect(img, confThreshold=CONFIDENCE_THRESHOLD, nmsThreshold=NMS_THRESHOLD)

    if classes == ():
        ready_img = img_path.parent / 'ready' / f'{img_path.stem}_ready{img_path.suffix}'
        ready_img.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(ready_img), img)
    else:
        for cls, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
            if float(confidence) > CONFIDENCE_THRESHOLD:
                current_class = NAMES[cls]
                current_thresh = float(confidence) * 100

                xmin_p = int(box[0])
                ymin_p = int(box[1])
                xmax_p = int(xmin_p + box[2])
                ymax_p = int(ymin_p + box[3])

                if xmin_p < 0:
                    xmin_p = 0

                if xmax_p > img.shape[1]:
                    xmax_p = img.shape[1]

                if ymin_p < 0:
                    ymin_p = 0

                if ymax_p > img.shape[0]:
                    ymax_p = img.shape[0]

                crop = img.copy()[ymin_p:ymax_p, xmin_p:xmax_p]

                cv2.rectangle(img, (xmin_p, ymin_p), (xmax_p, ymax_p), COLORS[cls], 1)
                text = f"{current_class} ({current_thresh:.1f}%)"
                fontType = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.35
                cv2.putText(
                    img=img,
                    text=text,
                    org=(xmin_p, ymin_p - 10),
                    fontFace=fontType,
                    fontScale=fontScale,
                    color=(0, 0, 255),
                    thickness=1,
                )  # print "Person"

    ready_img = img_path.parent / 'ready' / f'{img_path.stem}_ready{img_path.suffix}'
    ready_img.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ready_img), img)
