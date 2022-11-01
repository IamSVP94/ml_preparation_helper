'''
Script for creating train.txt and test.txt from dif dirs together
'''
from pathlib import Path
import random

TRAIN_DIRS = ['/home/vid/hdd/file/project/240-EVRAZ/data/attrs/train/']
TEST_DIRS = ['/home/vid/hdd/file/project/240-EVRAZ/data/attrs/test_base_dir',
             '/home/vid/hdd/file/project/240-EVRAZ/data/attrs/test_doobychenie/']

TRAIN_TXT_PATH =    '/home/vid/hdd/file/project/240-EVRAZ/cfg3110_2/train.txt'
TEST_TXT_PATH =     '/home/vid/hdd/file/project/240-EVRAZ/cfg3110_2/test.txt'


def get_filelist(dir_list):
    filelist = []
    for dir in dir_list:
        filelist.extend(list(map(str, (Path(dir).glob('**/*.jpg')))))
    return filelist


def make_txtfile(filelist, txtname, shuffle=True):
    if shuffle:
        random.seed(2)
        random.shuffle(filelist)
    with open(txtname, 'w') as wf:
        for idx, line in enumerate(filelist):
            wf.write(f'{line}\n')


trainfiles = get_filelist(TRAIN_DIRS)
testfiles = get_filelist(TEST_DIRS)

make_txtfile(trainfiles, TRAIN_TXT_PATH)
make_txtfile(testfiles, TEST_TXT_PATH)
