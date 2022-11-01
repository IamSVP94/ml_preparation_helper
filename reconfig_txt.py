'''
script for delete files in dirs from "del_dirs" from train file list
'''

FILE_PATH = '/home/vid/hdd/file/project/240-EVRAZ/cfg3110/train_for_pseudo.txt'
del_dirs = [
    '/home/vid/hdd/file/project/240-EVRAZ/data/attrs/train/Evraz_v3_attrs/',
    '/home/vid/hdd/file/project/240-EVRAZ/data/attrs/train/Evraz_v4_attrs/',
    '/home/vid/hdd/file/project/240-EVRAZ/data/attrs/train/Evraz_v5_attrs/',
]

with open(FILE_PATH, 'r') as rf:
    lines = rf.readlines()

with open(FILE_PATH, 'w') as wf:
    for idx, line in enumerate(lines):
        write_str_in_new_lines = True
        for del_dir in del_dirs:
            if del_dir in line and write_str_in_new_lines:
                write_str_in_new_lines = False
        else:
            if write_str_in_new_lines:
                wf.write(line)
