from pathlib import Path

import cv2
from tqdm import tqdm

IMG_DIR = '/home/vid/hdd/file/project/256_МешкиПодсчет/datasetv4/'
txts = list(Path(IMG_DIR).glob('*.txt'))
print(len(txts))

pbar = tqdm(txts)
for txt in pbar:
    with open(txt, 'r') as f:
        lines = [t.rstrip() for t in f.readlines()]
    if len(lines) > 1 and len(lines) != len(set(lines)):
        print(txt)
        print(lines)
        exit()
