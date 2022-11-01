import shutil
from pathlib import Path

from tqdm import tqdm

# mode = {'jpg': 'txt'}  # 'to': 'from'
mode = {'txt': 'jpg'}  # 'to': 'from'
TO_DIR = '/home/vid/hdd/file/project/240-EVRAZ/data/attrs/train/Evraz_v5_attrs/fresh/'
FROM_DIR = '/home/vid/hdd/file/project/240-EVRAZ/data/attrs/train/Evraz_v5_attrs/'

to_suffix = list(mode.keys())[0]
from_suffix = list(mode.values())[0]
readyfiles = list(Path(TO_DIR).glob(f'*.{to_suffix}'))

for ready_path in tqdm(readyfiles):
    copyfile_old_path = Path(FROM_DIR, f'{ready_path.stem}.{from_suffix}')
    copyfile_new_path = ready_path.with_suffix(f'.{from_suffix}')
    shutil.copy(copyfile_old_path, copyfile_new_path)
