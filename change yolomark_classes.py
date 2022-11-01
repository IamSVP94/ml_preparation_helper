from pathlib import Path

from tqdm import tqdm

DIR = '/home/vid/hdd/file/project/258-ЕврохимПломба/data/20221025_Пломбы/'

class_replacer = {
    1: 0,
}

txts = list(Path(DIR).glob('*.txt'))
pbar = tqdm(txts)
for txt_path in pbar:
    pbar.set_description(f'{txt_path}')
    with open(txt_path, 'r') as fr:
        lines = [t for t in fr.readlines()]
    for idx, line in enumerate(lines):
        cl = int(line[:2])
        if cl in class_replacer.keys():
            new_line = f'{class_replacer[cl]} {line[2:]}'
            lines[idx] = new_line
    with open(txt_path, 'w') as fw:
        fw.writelines(lines)
