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


class CrossBenchmark(Benchmark):
    def get_id_combination(self, cls=None, num=2):
        r"""
        Get the combination of images and length of combinations in specified class.

        :param cls: int or str, class of expected data. None for all classes
        :param num: int, number of images in each image ID list; for example, 2 for 2GM
        :return:
                id_combination_list: list of combinations of image ids

                length: length of combinations
        """

        if cls == None:
            clss = None
        elif type(cls) == str:
            clss = cls
        elif type(cls) == list:
            print('It is a list!')
            clss = cls

        with open(self.data_list_path) as f1:
            data_id = json.load(f1)

        length = 0
        id_combination_list = []

        if type(clss) == list:
            pass

        if clss != None:
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

        return id_combination_list, length
