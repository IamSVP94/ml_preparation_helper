import shutil
from pathlib import Path

from tqdm import tqdm

TXT_DIR = '/home/vid/hdd/file/project/240-EVRAZ/data/Evraz_v4.1_attrs/'
JPG_DIR = '/home/vid/hdd/file/project/240-EVRAZ/data/Evraz_v4.1_attrs_imgs/'

imgs = list(Path(JPG_DIR).glob('*.jpg'))
for img_path in tqdm(imgs):
    txt_old_path = Path(TXT_DIR, f'{img_path.stem}.txt')
    txt_new_path = img_path.with_suffix('.txt')
    shutil.copy(txt_old_path, txt_new_path)
