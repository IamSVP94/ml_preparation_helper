import shutil
from pathlib import Path

from tqdm import tqdm

IMG_DIR = '/home/vid/hdd/file/project/NLMK_SILA/datasets/attributes/data_minsize=(87,48)/NLMK_SILA_v25_attrs/data/img_3/'

classes = [10]
del_class_marks = True
classes_list_txt_path = '/home/vid/hdd/file/project/NLMK_SILA/datasets/attributes/data_minsize=(87,48)/10class/v25_img_3.txt'
new_dir = '/home/vid/hdd/file/project/NLMK_SILA/datasets/attributes/data_minsize=(87,48)/10class/v25_img_3/'

txts = list(Path(IMG_DIR).glob('**/*.txt'))
files = []
for txt in tqdm(txts):
    with open(txt, 'r') as f:
        for line in f.readlines():
            cl = int(line[:2])
            if cl in classes:
                files.append(txt)
                continue

with open(classes_list_txt_path, 'w') as f:
    Path(new_dir).mkdir(exist_ok=True, parents=True)
    for file in set(files):
        img_path = file.with_suffix('.jpg')
        # print(img_path, img_path.exists())
        new_img_path = Path(new_dir, img_path.name)
        new_txt_path = Path(new_dir, file.name)

        with open(file, 'r') as fr:
            lines = fr.readlines()
        new_txt_content = []

        if del_class_marks == True:
            new_txt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(new_txt_path, 'w') as fw:
                for line in lines:
                    if int(line[:2]) not in classes:
                        fw.write(line)
        else:
            shutil.copy(file, new_txt_path)

        shutil.copy(img_path, new_img_path)
        if new_img_path.exists():
            f.write(f'{str(new_img_path)}\n')
    else:
        print(classes_list_txt_path)
        print(new_dir, len(set(files)))
