from pygmtools.benchmark import Benchmark
import tempfile
import os
import shutil
from PIL import Image
import numpy as np
import random
import json
import itertools
from scipy.sparse import coo_matrix
from pygmtools.dataset import *
import csv


class CrossBenchmark(Benchmark):
    """def __init__(self, name, sets, obj_resize=(256, 256), problem='2GM', filter='intersection', classes=None, **args):
        super().__init__(name, sets, obj_resize, problem, filter, **args)
        self.classes = [classes]"""

    def get_id_combination(self, cls=None, num=2):
        r"""
        Get the combination of images and length of combinations in specified class.

        :param cls: int or str, class of expected data. None for all classes
        :param num: int, number of images in each image ID list; for example, 2 for 2GM
        :return:
                id_combination_list: list of combinations of image ids

                length: length of combinations
        """
        # print(f'cls: {cls}  type:{type(cls)}')
        # print('\nget_id_combinations')
        if cls == None:
            clss = None
        elif type(cls) == str:
            clss = cls

        with open(self.data_list_path) as f1:
            data_id = json.load(f1)
        length = 0
        id_combination_list = []

        if clss.count('_') > 0:
            class1, class2 = clss.split('_')
            #print(f'class1: {class1}   class2: {class2}')
            data_list1 = []
            data_list2 = []
            if self.name != 'SPair71k':
                for id in data_id:
                    # print(f"id: {id}") #id: 2008_004259_1_tvmonitor (comes from train.json)
                    if self.data_dict[id]['cls'] == class1:
                        data_list1.append(id)
                    elif self.data_dict[id]['cls'] == class2:
                        data_list2.append(id)
                id_combination = [(i, j) for i in data_list1 for j in data_list2]
                # id_combination = list(itertools.combinations(data_list, num))
                length += len(id_combination)
                id_combination_list.append(id_combination)

        elif clss != None:
            data_list = []
            if self.name != 'SPair71k':
                for id in data_id:
                    # print(f"id: {id}") #id: 2008_004259_1_tvmonitor
                    if self.data_dict[id]['cls'] == clss:
                        data_list.append(id)
                id_combination = list(itertools.combinations(data_list, num))
                length += len(id_combination)
                id_combination_list.append(id_combination)
            else:
                for id_pair in data_id:
                    if self.data_dict[id_pair[0]]['cls'] == clss:
                        data_list.append(id_pair)
                length += len(data_list)
                id_combination_list.append(data_list)
        else:
            for clss in self.classes:       # for all classes
                data_list = []
                if self.name != 'SPair71k':
                    for id in data_id:
                        if self.data_dict[id]['cls'] == clss:
                            data_list.append(id)
                    id_combination = list(itertools.combinations(data_list, num))
                    length += len(id_combination)
                    id_combination_list.append(id_combination)
                else:
                    for id_pair in data_id:
                        if self.data_dict[id_pair[0]]['cls'] == clss:
                            data_list.append(id_pair)
                    length += len(data_list)
                    id_combination_list.append(data_list)
        with open('pairs', 'w') as f:
            write = csv.writer(f)
            write.writerows(id_combination_list)
        return id_combination_list, length

    def compute_length(self, cls=None, num=2):
        r"""
        Compute the length of image combinations in specified class.

        :param cls: int or str, class of expected data. None for all classes
        :param num: int, number of images in each image ID list; for example, 2 for 2GM
        :return: length of combinations
        """
        # print(f'\ncompute_length')
        if cls == None:
            clss = None
        elif type(cls) == str:
            clss = cls

        with open(self.data_list_path) as f1:
            data_id = json.load(f1)

        length = 0
        #print(f'cls: {cls}')
        if clss.count('_') > 0:
            #print(f'\n\n\ncompute_length:\nclss: {clss}')
            class1, class2 = clss.split('_')

            data_list1 = []
            data_list2 = []
            if self.name != 'SPair71k':
                for id in data_id:
                    # print(f"id: {id}") #id: 2008_004259_1_tvmonitor (comes from train.json)
                    if self.data_dict[id]['cls'] == class1:
                        data_list1.append(id)
                    elif self.data_dict[id]['cls'] == class2:
                        data_list2.append(id)
                id_combination = [(i, j) for i in data_list1 for j in data_list2]
                # id_combination = list(itertools.combinations(data_list, num))
                length += len(id_combination)

        if clss != None:
            if self.name != 'SPair71k':
                data_list = []
                for id in data_id:
                    if self.data_dict[id]['cls'] == clss:
                        data_list.append(id)
                id_combination = list(itertools.combinations(data_list, num))
                length += len(id_combination)
            else:
                for id_pair in data_id:
                    if self.data_dict[id_pair[0]]['cls'] == clss:
                        length += 1

        else:
            for clss in self.classes:
                if self.name != 'SPair71k':
                    data_list = []
                    for id in data_id:
                        if self.data_dict[id]['cls'] == clss:
                            data_list.append(id)
                    id_combination = list(itertools.combinations(data_list, num))
                    length += len(id_combination)
                else:
                    for id_pair in data_id:
                        if self.data_dict[id_pair[0]]['cls'] == clss:
                            length += 1
        # print(f'length: {length}')
        # print(f'len(datalist_1): {len(data_list1)}')
        # print(f'len(datalist_2): {len(data_list2)}')
        return length

    def compute_img_num(self, classes):
        r"""
        Compute number of images in specified classes.

        :param classes: list of dataset classes
        :return: list of numbers of images in each class
        """
        # print("\ncompute_img_num")
        with open(self.data_list_path) as f1:
            data_id = json.load(f1)
        num_list = []
        for clss in classes:   ## idk why
            cls_img_num = 0
            if self.name != 'SPair71k':
                #print(f'clss: {clss}')
                class1, class2 = clss.split('_')
                #print(f'class1: {class1}   class2: {class2}')
                for id in data_id:
                    if self.data_dict[id]['cls'] == class1:
                        cls_img_num += 1
                    elif self.data_dict[id]['cls'] == class2:
                        cls_img_num += 1
                num_list.append(cls_img_num)
            else:
                img_cache = []
                for id_pair in data_id:
                    if self.data_dict[id_pair[0]]['cls'] == clss:
                        if id_pair[0] not in img_cache:
                            img_cache.append(id_pair[0])
                            cls_img_num += 1
                        if id_pair[1] not in img_cache:
                            img_cache.append(id_pair[1])
                            cls_img_num += 1
                num_list.append(cls_img_num)
        # print(f'num_list: {num_list}')
        return num_list

    def rand_get_data(self, cls=None, num=2, test=False, shuffle=True):
        r"""
        Randomly fetch data for training or test. Implemented by calling ``get_data`` function.

        :param cls: int or str, class of expected data. None for random class
        :param num: int, number of images; for example, 2 for 2GM
        :param test: bool, whether the fetched data is used for test; if true, this function will not return ground truth
        :param shuffle: bool, whether to shuffle the order of keypoints
        :return:
                    **data_list**: list of data, like ``[{'img': np.array, 'kpts': coordinates of kpts}, ...]``

                    **perm_mat_dict**: ground truth, like ``{(0,1):scipy.sparse, (0,2):scipy.sparse, ...}``, ``(0,1)`` refers to data pair ``(ids[0],ids[1])``

                    **ids**: list of image ID
        """
        # print("\n\n\nrand_get_data")
        # print(f"\ncls: {cls}")
        if cls == None:
            cls = random.randrange(0, len(self.classes))
            clss = self.classes[cls]
        elif type(cls) == str:
            clss = cls

        with open(self.data_list_path) as f1:
            data_id = json.load(f1)

        data_list = []
        ids = []
        if self.name != 'SPair71k':
            if clss.count('_')>0:
                class1, class2 = clss.split('_')
                for id in data_id:
                    if self.data_dict[id]['cls'] == class1 or self.data_dict[id]['cls'] == class2:
                        data_list.append(id)

                for objID in random.sample(data_list, num):
                    ids.append(objID)
            else:

                for id in data_id:
                    if self.data_dict[id]['cls'] == clss:
                        data_list.append(id)

                for objID in random.sample(data_list, num):
                    ids.append(objID)
        else:
            for id in data_id:
                if self.data_dict[id[0]]['cls'] == clss:
                    data_list.append(id)
            ids = random.sample(data_list, 1)[0]

        return self.get_data(ids, test, shuffle)
