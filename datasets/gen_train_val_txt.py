import os
import glob
from random import shuffle

def scan_folder(path, key):
    key_words = '*' + key
    file_list = glob.glob(os.path.join(path, key_words))
    return file_list

def random_file(file_list):
    shuffle(file_list)
    size = len(file_list)

    train_size = round(size * 0.8)

    train_file = file_list[:train_size]
    val_file = file_list[train_size:]
    return train_file, val_file
def image_file_write(file_list, file_name, pwd):
    fp = open(file_name, 'w')
    for file in file_list:
        file = pwd + file[1:]
        fp.write(file+'\n')
    fp.close()

file_list = scan_folder('./bus_passenger', '.jpg')
#print(file_list, len(file_list))
train_file, val_file = random_file(file_list)
#print(len(train_file), len(val_file))
pwd = os.path.abspath(os.path.dirname(__file__))
#print(pwd)
image_file_write(train_file, "bus_passenger_train.txt", pwd)
image_file_write(val_file, "bus_passenger_val.txt", pwd)
