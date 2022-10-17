from pathlib import Path

IMG_DIR = '/home/vid/hdd/file/project/256_МешкиПодсчет/dataset/'
# IMG_DIR = '/home/vid/hdd/file/project/256_МешкиПодсчет/razmetka10102022/Elvira/'
CLASSES_NAMES = ['bag']

CFG_DIR = Path('/home/vid/hdd/file/project/256_МешкиПодсчет/cfgv7/')
FILE_LIST_NAME = CFG_DIR / 'full.txt'
OBJ_NAMES_NAME = CFG_DIR / 'obj.names'

imgs = []
for format in ['jpg', 'png']:
    imgs.extend(list(Path(IMG_DIR).glob(f'**/*.{format}')))
print(len(imgs))

CFG_DIR.mkdir(parents=True, exist_ok=True)
with open(FILE_LIST_NAME, 'w') as txt:
    txt.writelines(map(lambda path: f'{path}\n', imgs))
with open(OBJ_NAMES_NAME, 'w') as txt:
    txt.writelines(map(lambda path: f'{path}\n', CLASSES_NAMES))
