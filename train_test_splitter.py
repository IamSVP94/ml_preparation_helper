import shutil
import subprocess
from pathlib import Path

from sklearn.model_selection import train_test_split

CFG_DIR = Path('/home/vid/hdd/file/project/256_МешкиПодсчет/cfgv7/')

FULL_TXT_PATH = Path(CFG_DIR) / 'full.txt'
NAMES_PATH = Path(FULL_TXT_PATH).parent / 'obj.names'
width, height = 512, 512
test_size = 0.2

# new files path
cfg_dir = Path(FULL_TXT_PATH).parent
TRAIN_TXT_PATH = cfg_dir / 'train.txt'
TEST_TXT_PATH = cfg_dir / 'test.txt'
DATA_PATH = cfg_dir / 'obj.data'
with open(FULL_TXT_PATH, 'r') as txt:
    lines = txt.readlines()

# make train.txt and test.txt
train, test = train_test_split(lines, test_size=test_size, shuffle=True, random_state=2)
assert len(train) + len(test) == len(lines)
with open(TRAIN_TXT_PATH, 'w') as txt:
    txt.writelines(map(lambda path: f'{path}', train))
with open(TEST_TXT_PATH, 'w') as txt:
    txt.writelines(map(lambda path: f'{path}', test))

# copy weight and congig to train dir
weights_path = '/home/vid/hdd/projects/PycharmProjects/yolomark_data/yolov4.conv.137'
cfg_file_path = '/home/vid/hdd/projects/PycharmProjects/yolomark_data/yolov4-custom.cfg'
new_weights_path = cfg_dir / Path(weights_path).name
new_cfg_file_path = cfg_dir / Path(cfg_file_path).name
shutil.copy(weights_path, new_weights_path)
shutil.copy(cfg_file_path, new_cfg_file_path)

# make obj.data
DATA_STRINGS = [
    f'classes={len(open(NAMES_PATH, "r").readlines())}',
    f'train={TRAIN_TXT_PATH}',
    f'valid={TEST_TXT_PATH}',
    f'names={NAMES_PATH}',
    f'backup=backup/',
]
with open(DATA_PATH, 'w') as txt:
    txt.writelines(map(lambda path: f'{path}\n', DATA_STRINGS))

# mare train_run.sh file
obj_data_path = "/".join(DATA_PATH.parts[-2:])
with open(DATA_PATH.parent.parent / 'run_train.sh', 'w') as txt:
    command = f'darknet detector train {obj_data_path} {new_cfg_file_path} {new_weights_path}'
    txt.write(command)

# calc anchors for detector
# command = f'darknet detector calc_anchors {DATA_PATH} -num_of_clusters 9 -width {width} -height {height}'
# print(command)
# subprocess.run(str(command), shell=True)
