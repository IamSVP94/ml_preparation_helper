import shutil
import subprocess
from pathlib import Path
from sklearn.model_selection import train_test_split
from txt_maker import CFG_DIR, IMG_DIR

mode = 'move'  # copy or move
CFG_DIR, IMG_DIR = Path(CFG_DIR), Path(IMG_DIR)
FULL_TXT_PATH = Path(CFG_DIR) / 'full.txt'
NAMES_PATH = Path(FULL_TXT_PATH).parent / 'obj.names'
test_size = 0.2
# width, height = 512, 512

# new files path
cfg_dir = Path(FULL_TXT_PATH).parent
DATA_PATH = cfg_dir / 'obj.data'
with open(FULL_TXT_PATH, 'r') as txt:
    lines = txt.readlines()

# make train.txt and test.txt
train, test = train_test_split(lines, test_size=test_size, shuffle=True, random_state=2)
assert len(train) + len(test) == len(lines)
datasets_dirs = {
    'train': {
        'dataset': train,
        'txt_path': cfg_dir / 'train.txt',
        'new_files_dir_path': 'train/',
        'pathlist': [],
    },
    'test': {
        'dataset': test,
        'txt_path': cfg_dir / 'test.txt',
        'new_files_dir_path': 'test_base_dir/test1/',  # 'test_doobychenie/test_doobychenie1'
        'pathlist': [],
    },
}

# make train.txt and test.txt with file move in train test dirs
ALL_FILES_PATHES = []
for datasets_dirname, dataset_meta in datasets_dirs.items():
    full_datasets_dirname = IMG_DIR / f'{dataset_meta["new_files_dir_path"]}'
    for element_path in dataset_meta['dataset']:
        element_path = Path(element_path.strip('\n'))
        parent_dir_parts = IMG_DIR.parts
        last_part = '/'.join(element_path.parts[len(parent_dir_parts):])
        new_elementa_path = full_datasets_dirname / last_part  # new file path
        new_elementa_path.parent.mkdir(parents=True, exist_ok=True)  # make parent dir for file (or check for exist)
        if mode == 'move':
            shutil.move(element_path, new_elementa_path)  # move file
        elif mode == 'copy':
            shutil.copy(element_path, new_elementa_path)  # copy file
        txt_path = element_path.with_suffix('.txt')
        if txt_path.exists():
            new_txt_path = new_elementa_path.with_suffix('.txt')  # new markup file path
            if mode == 'move':
                shutil.move(txt_path, new_txt_path)  # move markup file
            elif mode == 'copy':
                shutil.copy(txt_path, new_txt_path)  # copy markup file
        dataset_meta['pathlist'].append(str(new_elementa_path))  # add new path into .txt file
    else:
        ALL_FILES_PATHES.extend(dataset_meta['pathlist'])
        TXT_PATH = datasets_dirs[datasets_dirname]['txt_path']
        with open(TXT_PATH, 'w') as txt:
            txt.writelines(map(lambda path: f'{path}\n', dataset_meta['pathlist']))
else:
    with open(FULL_TXT_PATH, 'w') as txt:
        txt.writelines(map(lambda path: f'{path}\n', ALL_FILES_PATHES))

# copy weight and congig to cfg dir
weights_path = '/home/vid/hdd/projects/PycharmProjects/yolomark_data/yolov4.conv.137'
cfg_file_path = '/home/vid/hdd/projects/PycharmProjects/yolomark_data/yolov4-custom.cfg'
new_weights_path = cfg_dir / Path(weights_path).name
new_cfg_file_path = cfg_dir / Path(cfg_file_path).name
shutil.copy(weights_path, new_weights_path)
shutil.copy(cfg_file_path, new_cfg_file_path)

# make obj.data
DATA_STRINGS = [
    f'classes={len(open(NAMES_PATH, "r").readlines())}',
    f'train={datasets_dirs["train"]["txt_path"]}',
    f'valid={datasets_dirs["test"]["txt_path"]}',
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
