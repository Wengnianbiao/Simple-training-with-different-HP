#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2020/6/5 19:30
# @Author: Biao
# @File: genarate_val_datasets
import numpy as np
import os
import shutil
"""
Separate the source dataset into train data and test data(default 7:3)
The architecture of source dataset should be one folder for each category.
"""


class GetDatasets(object):
    def __init__(self,path,classes=10,train_rate=0.7):
        """
        根据特定数据集将其分割成指定类别的ImageFolder格式:
                            --> train --> classes个以类别标签命名的文件夹(每个文件夹对应类别的训练图片)
        datasets_name_folder-
                            --> test --> classes个以类别标签命名的文件夹(每个文件夹对应类别的测试图片)
        :param path: datasets path.
        :param classes: the number of categorise you want to retrieve.
        :param train_rate: proportion of training image.
        """
        self.path = path
        self.classes = classes
        self.train_rate = train_rate

    def sep_datasets(self):
        # Create folders for training data and test data.
        new_path = ('./data' + str(self.classes) + '/train', './data' + str(self.classes) + '/test')

        # According to user's fixed classes,choose the numbers of classes categories randomly.
        class_array = np.array(np.random.choice(len(os.listdir(path)), self.classes, replace=False))
        class_array = np.array(os.listdir(path))[class_array]
        for f in class_array:  # Iterate through each category.
            for p in new_path:
                if not os.path.exists(os.path.join(p, f)):
                    os.makedirs(os.path.join(p, f))

            upd_path = os.path.join(path, f)
            len_cate = len(os.listdir(upd_path))
            # Choice randomly number of current category according to the train_rate.
            # The value of parameter replace should be set False to ensure the list does't contain duplicate elements.
            train_list = np.random.choice(len_cate, int(self.train_rate * len_cate), replace=False)
            # If the number is in the train_list,copy the image into train folder.
            for ind, img in enumerate(os.listdir(upd_path)):
                if ind in train_list:
                    shutil.copy(os.path.join(upd_path, img), os.path.join(new_path[0], f))
                else:
                    shutil.copy(os.path.join(upd_path, img), os.path.join(new_path[1], f))


if __name__ == '__main__':
    path = './101_ObjectCategories'
    Data = GetDatasets(path,classes=20,train_rate=0.8)
    Data.sep_datasets()
