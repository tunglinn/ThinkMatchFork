import torch.cuda
import torch.optim as optim
import time
import xlwt
from datetime import datetime
from pathlib import Path
from tensorboardX import SummaryWriter

from src.dataset.data_loader_cross import GMDataset, get_dataloader     # using self defined data_loader_cross class
from src.displacement_layer import Displacement
from src.loss_func import *
from src.evaluation_metric import matching_accuracy
from src.parallel import DataParallel
from src.utils.model_sl import load_model, save_model
from eval import eval_model
from src.lap_solvers.hungarian import hungarian
from src.utils.data_to_cuda import data_to_cuda

from src.utils.config import cfg
from pygmtools.benchmark import Benchmark
from cross_benchmark import CrossBenchmark
from render_keypoints import render_one, render_pair

from src.utils.dup_stdout_manager import DupStdoutFileManager
from src.utils.parse_args import parse_args
from src.utils.print_easydict import print_easydict

args = parse_args('Deep learning of graph matching training & evaluation code.')

import importlib

mod = importlib.import_module(cfg.MODULE)
Net = mod.Net

torch.manual_seed(cfg.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(cfg.RANDOM_SEED)


dataset_len = {'train': cfg.TRAIN.EPOCH_ITERS * cfg.BATCH_SIZE, 'test': cfg.EVAL.SAMPLES}
ds_dict = cfg[cfg.DATASET_FULL_NAME] if ('DATASET_FULL_NAME' in cfg) and (cfg.DATASET_FULL_NAME in cfg) else {}
# print(ds_dict)


benchmark = {
        x: CrossBenchmark(name=cfg.DATASET_FULL_NAME,
                          sets=x,
                          problem=cfg.PROBLEM.TYPE,
                          obj_resize=cfg.PROBLEM.RESCALE,
                          filter=cfg.PROBLEM.FILTER,
                          classes=cfg.TRAIN.CLASS,
                          **ds_dict)
        for x in ('train', 'test')}

image_dataset = {
        x: GMDataset(cfg.DATASET_FULL_NAME,
                     benchmark[x],
                     dataset_len[x],
                     cfg.PROBLEM.TRAIN_ALL_GRAPHS if x == 'train' else cfg.PROBLEM.TEST_ALL_GRAPHS,
                     cfg.TRAIN.CLASS if x == 'train' else cfg.EVAL.CLASS,
                     cfg.PROBLEM.TYPE)
        for x in ('train', 'test')}

# print(image_dataset['train'].bm.get_id_combination('cat_chair'))

dataloader = {x: get_dataloader(image_dataset[x], shuffle=True, fix_seed=(x == 'test')) for x in ('train', 'test')}

exit(1)
print('\n\n\nRender pair.')
for inputs in dataloader['test']:
    pairs = inputs['id_list']
    print(f'Pairs are: \n{pairs[0]}\n{pairs[1]}')
    # render_pair(pairs[0][0], pairs[1][0])
