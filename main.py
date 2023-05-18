from pygmtools.benchmark import Benchmark
from src.utils.config import cfg
from src.dataset.data_loader import GMDataset, get_dataloader

dataset_len = {'train': cfg.TRAIN.EPOCH_ITERS * cfg.BATCH_SIZE, 'test': cfg.EVAL.SAMPLES}
ds_dict = cfg[cfg.DATASET_FULL_NAME] if ('DATASET_FULL_NAME' in cfg) and (cfg.DATASET_FULL_NAME in cfg) else {}
print(ds_dict)

"""
benchmark = {
        x: Benchmark(name=cfg.DATASET_FULL_NAME,
                     sets=x,
                     problem=cfg.PROBLEM.TYPE,
                     obj_resize=cfg.PROBLEM.RESCALE,
                     filter=cfg.PROBLEM.FILTER,
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
dataloader = {x: get_dataloader(image_dataset[x], shuffle=True, fix_seed=(x == 'test')) for x in ('train', 'test')}"""
