import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class bus_passenger(Dataset):
    def __init__(self, mode):
        ## 可根据实际情况修改后面的实际文件位置
        if mode == 'train':
            self.file_name = './datasets/bus_passenger_train.txt'
        elif mode == 'val':
            self.file_name = './datasets/bus_passenger_val.txt'
        else:
            print("mode is train or val")
            exit(-1)
        ## 后续需要根据实际读入进行修改
        self.image_width = 320
        self.image_height = 240
        self.image_list, self.label_list = self.__image_and_label_list_gen()
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.__get_image(self.image_list[idx])
        label = self.__get_label(self.label_list[idx])
        return image, label

    def  __image_and_label_list_gen(self):
        image_list = []
        label_list = []
        if os.path.exists(self.file_name) == False:
            print(self.file_name, " is not existed\n")
            exit(-1)
        fp = open(self.file_name, 'r')
        contents = fp.readlines()
        for info in contents:
            info = info.strip('\n')
            image_list.append(info)
            label_info = info[:-3] + 'txt'
            label_list.append(label_info)
        fp.close()
        #print(image_list, label_list)
        return image_list, label_list
    def __get_image(self, file_name):
        image = cv2.imread(file_name)
        image = cv2.resize(image, (300, 300))
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image.float()
    def __get_label(self, file_name):
        fp = open(file_name, 'r')
        contents = fp.readlines()
        gt_list = np.zeros((0, 4))
        for info in contents:
            info = info.strip('\n')
            infos = info.split(' ')
            gt = np.zeros((1, 4))
            ## 按照实际需求输出
            for i, str_num in enumerate(infos):
                if i == 2 or i == 4:
                    gt[0, i-1] = float(str_num) * self.image_height / 300.0
                elif i == 1 or i == 3:
                    gt[0, i-1] = float(str_num) * self.image_width / 300.0
            if gt[0, 2] != 0 and gt[0, 3] != 0:
                #gt[0] = gt[0] - int(gt[2]/2)
                #gt[1] = gt[1] - int(gt[3]/2)
                gt[0, 2] = gt[0, 0] + gt[0, 2]
                gt[0, 3] = gt[0, 1] + gt[0, 3]
                gt = np.array(gt)
                gt_list = np.append(gt_list, gt, axis=0)
        fp.close()
        #print(gt_list.shape)
        gt_list = torch.from_numpy(gt_list).float()
        return gt_list


# train_data = bus_passenger('train')
# length = train_data.__len__()
# for i in range(0, length):
#     image, label = train_data.__getitem__(i)
#     print(label)
#     color = (0, 255, 0) # BGR
#     for one_label in label:
#         cv2.rectangle(image, (one_label[0], one_label[1]), \
#         (one_label[2], one_label[3]), color)
#     cv2.imshow("image", image)
#     cv2.waitKey(0)