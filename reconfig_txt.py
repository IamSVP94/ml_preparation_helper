'''
script for delete files in dirs from "del_dirs" from train file list
'''

FILE_PATH = '/home/vid/hdd/file/project/256_МешкиПодсчет/cfgv6_2/train.txt'
del_dirs = ['/home/vid/hdd/file/project/256_МешкиПодсчет/dataset/img_5/']

with open(FILE_PATH, 'r') as rf:
    lines = rf.readlines()

with open(FILE_PATH, 'w') as wf:
    for line in lines:
        write_str_in_new_lines = True
        for del_dir in del_dirs:
            if del_dir in line:
                write_str_in_new_lines = False
                continue
            if write_str_in_new_lines == False:
                continue
            else:
                wf.write(line)
