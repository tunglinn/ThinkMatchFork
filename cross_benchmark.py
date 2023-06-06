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
import pandas as pd


class CrossBenchmark(Benchmark):
    """used to generate cross category datasets like pairs of one cat and one chair

        def __init__(self, name, sets, obj_resize=(256, 256), problem='2GM', filter='intersection', classes=None, **args):
        super().__init__(name, sets, obj_resize, problem, filter, **args)
        self.classes = [classes]"""

    def __init__(self, name, sets, obj_resize=(256, 256), problem='2GM', filter='intersection', **args):
        super().__init__(name, sets, obj_resize, problem, filter, **args)
        self.cross_match_file = 'match_test.csv'
        self.cross_match_path = os.path.join(self.cross_match_file)
        self.cross_matchings = self.get_cross_matchings()

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
        elif type(cls) == str: # also takes in cross string ex. 'cat_chair'
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
                for id in data_id: #id: 2008_004259_1_tvmonitor (comes from train.json)
                    # print(f"id: {id}")
                    if self.data_dict[id]['cls'] == class1:     # find ids from class1
                        data_list1.append(id)
                    elif self.data_dict[id]['cls'] == class2:   # find ids from class2
                        data_list2.append(id)
                id_combination = [(i, j) for i in data_list1 for j in data_list2]   # combines two lists, all combinations
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

    def get_data(self, ids, test=False, shuffle=True):
        r"""
        Fetch a data pair or pairs of data by image ID for training or test.

        :param ids: list of image ID, usually in ``train.json`` or ``test.json``
        :param test: bool, whether the fetched data is used for test; if true, this function will not return ground truth
        :param shuffle: bool, whether to shuffle the order of keypoints
        :return:
                    **data_list**: list of data, like ``[{'img': np.array, 'kpts': coordinates of kpts}, ...]``

                    **perm_mat_dict**: ground truth, like ``{(0,1):scipy.sparse, (0,2):scipy.sparse, ...}``, ``(0,1)`` refers to data pair ``(ids[0],ids[1])``

                    **ids**: list of image ID
        """
        assert (self.problem == '2GM' and len(ids) == 2) or ((self.problem == 'MGM' or self.problem == 'MGM3') and len(
            ids) > 2), '{} problem cannot get {} data'.format(self.problem, len(ids))

        ids.sort()
        data_list = []
        for keys in ids:
            obj_dict = dict()
            boundbox = self.data_dict[keys]['bounds']
            img_file = self.data_dict[keys]['path']
            with Image.open(str(img_file)) as img:
                obj = img.resize(self.obj_resize, resample=Image.BICUBIC,
                                 box=(boundbox[0], boundbox[1], boundbox[2], boundbox[3]))
                if self.name == 'CUB2011':
                    if not obj.mode == 'RGB':
                        obj = obj.convert('RGB')
            obj_dict['img'] = np.array(obj)
            obj_dict['kpts'] = self.data_dict[keys]['kpts']
            obj_dict['cls'] = self.data_dict[keys]['cls']
            obj_dict['univ_size'] = self.data_dict[keys]['univ_size']
            if shuffle:
                random.shuffle(obj_dict['kpts'])
            data_list.append(obj_dict)

        perm_mat_dict = dict()
        id_combination = list(itertools.combinations(list(range(len(ids))), 2))

        # change so that cross category keypoints map to each other
        # for example, cat: L_F_Elbow --> chair: Leg_Left_Front
        for id_tuple in id_combination:
            # creates img1[keypoint len] by img2[keypoint len] matrix
            perm_mat = np.zeros([len(data_list[_]['kpts']) for _ in id_tuple], dtype=np.float32)
            row_list = []
            col_list = []

            for i, keypoint in enumerate(data_list[id_tuple[0]]['kpts']):
                for j, _keypoint in enumerate(data_list[id_tuple[1]]['kpts']):
                    if keypoint['labels'] == _keypoint['labels']:
                        row_list.append(i)
                        col_list.append(j)
                        if keypoint['labels'] != 'outlier':
                            # if same keypoint, set to 1
                            perm_mat[i, j] = 1

            # I think we can just add these to the first for loop
            """for i, keypoint in enumerate(data_list[id_tuple[0]]['kpts']):
                for j, _keypoint in enumerate(data_list[id_tuple[1]]['kpts']):
                    if keypoint['labels'] == _keypoint['labels']:
                        row_list.append(i)
                        break
            for i, keypoint in enumerate(data_list[id_tuple[1]]['kpts']):
                for j, _keypoint in enumerate(data_list[id_tuple[0]]['kpts']):
                    if keypoint['labels'] == _keypoint['labels']:
                        col_list.append(i)
                        break"""
            row_list.sort()
            col_list.sort()
            if self.filter == 'intersection':
                perm_mat = perm_mat[row_list, :]
                perm_mat = perm_mat[:, col_list]
                data_list[id_tuple[0]]['kpts'] = [data_list[id_tuple[0]]['kpts'][i] for i in row_list]
                data_list[id_tuple[1]]['kpts'] = [data_list[id_tuple[1]]['kpts'][i] for i in col_list]
            elif self.filter == 'inclusion':
                perm_mat = perm_mat[row_list, :]
                data_list[id_tuple[0]]['kpts'] = [data_list[id_tuple[0]]['kpts'][i] for i in row_list]
            if not (len(ids) > 2 and self.filter == 'intersection'):
                sparse_perm_mat = coo_matrix(perm_mat)
                perm_mat_dict[id_tuple] = sparse_perm_mat

        # since we're just comparing 2 graphs, this will probably never run
        if len(ids) > 2 and self.filter == 'intersection':
            for p in range(len(ids) - 1):
                perm_mat_list = [np.zeros([len(data_list[p]['kpts']), len(x['kpts'])], dtype=np.float32) for x in
                                 data_list[p + 1: len(ids)]]
                row_list = []
                col_lists = []
                for i in range(len(ids) - p - 1):
                    col_lists.append([])

                for i, keypoint in enumerate(data_list[p]['kpts']):
                    kpt_idx = []
                    for anno_dict in data_list[p + 1: len(ids)]:
                        kpt_name_list = [x['labels'] for x in anno_dict['kpts']]
                        if keypoint['labels'] in kpt_name_list:
                            kpt_idx.append(kpt_name_list.index(keypoint['labels']))
                        else:
                            kpt_idx.append(-1)
                    row_list.append(i)
                    for k in range(len(ids) - p - 1):
                        j = kpt_idx[k]
                        if j != -1:
                            col_lists[k].append(j)
                            if keypoint['labels'] != 'outlier':
                                perm_mat_list[k][i, j] = 1

                row_list.sort()
                for col_list in col_lists:
                    col_list.sort()

                for k in range(len(ids) - p - 1):
                    perm_mat_list[k] = perm_mat_list[k][row_list, :]
                    perm_mat_list[k] = perm_mat_list[k][:, col_lists[k]]
                    id_tuple = (p, k + p + 1)
                    perm_mat_dict[id_tuple] = coo_matrix(perm_mat_list[k])

        if self.sets == 'test':
            for pair in id_combination:
                id_pair = (ids[pair[0]], ids[pair[1]])
                gt_path = os.path.join(self.gt_cache_path, str(id_pair) + '.npy')
                if not os.path.exists(gt_path):
                    np.save(gt_path, perm_mat_dict[pair])

        if not test:
            return data_list, perm_mat_dict, ids
        else:
            return data_list, ids