import argparse
import datetime
import logging
import math
import random
import time
import torch
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (
    MessageLogger,
    check_resume,
    get_env_info,
    get_root_logger,
    get_time_str,
    init_tb_logger,
    init_wandb_logger,
    make_exp_dirs,
    mkdir_and_rename,
    set_random_seed,
)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse

import numpy as np


def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, required=True, help="Path to option YAML file.")
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # distributed settings
    if args.launcher == "none":
        opt["dist"] = False
        print("Disable distributed.", flush=True)
    else:
        opt["dist"] = True
        if args.launcher == "slurm" and "dist_params" in opt:
            init_dist(args.launcher, **opt["dist_params"])
        else:
            init_dist(args.launcher)
            print("init dist .. ", args.launcher)

    opt["rank"], opt["world_size"] = get_dist_info()

    # random seed
    seed = opt.get("manual_seed")
    if seed is None:
        seed = random.randint(1, 10000)
        opt["manual_seed"] = seed
    set_random_seed(seed + opt["rank"])

    return opt


def init_loggers(opt):
    log_file = osp.join(opt["path"]["log"], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name="basicsr", log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize wandb logger before tensorboard logger to allow proper sync:
    if (
        (opt["logger"].get("wandb") is not None)
        and (opt["logger"]["wandb"].get("project") is not None)
        and ("debug" not in opt["name"])
    ):
        assert opt["logger"].get("use_tb_logger") is True, "should turn on tensorboard when using wandb"
        init_wandb_logger(opt)
    tb_logger = None
    if opt["logger"].get("use_tb_logger") and "debug" not in opt["name"]:
        tb_logger = init_tb_logger(log_dir=osp.join("tb_logger", opt["name"]))
    return logger, tb_logger


# PrefetchDateLoader，DataLoader等诸多配置参数
def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":  # train字段
            dataset_enlarge_ratio = dataset_opt.get("dataset_enlarge_ratio", 1)  # 未设置则为default值1
            train_set = create_dataset(dataset_opt)  # name:如Restormer,model_type:如ImageCleanModel

            # train_smpler
            train_sampler = EnlargedSampler(train_set, opt["world_size"], opt["rank"], dataset_enlarge_ratio)
            # train_loader
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt["num_gpu"],
                dist=opt["dist"],
                sampler=train_sampler,
                seed=opt["manual_seed"],
            )

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt["batch_size_per_gpu"] * opt["world_size"])
            )  # 计算epoch_periter
            total_iters = int(opt["train"]["total_iter"])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info(
                "Training statistics:"
                f"\n\tNumber of train images: {len(train_set)}"
                f"\n\tDataset enlarge ratio: {dataset_enlarge_ratio}"
                f"\n\tBatch size per gpu: {dataset_opt['batch_size_per_gpu']}"
                f"\n\tWorld size (gpu number): {opt['world_size']}"
                f"\n\tRequire iter number per epoch: {num_iter_per_epoch}"
                f"\n\tTotal epochs: {total_epochs}; iters: {total_iters}."
            )
        ##########
        #####
        ###
        elif phase == "val":  # val字段
            val_set = create_dataset(dataset_opt)
            # val_loader
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt["num_gpu"],
                dist=opt["dist"],
                sampler=None,
                seed=opt["manual_seed"],
            )
            logger.info(f"Number of val images/folders in {dataset_opt['name']}: {len(val_set)}")
        else:
            raise ValueError(f"Dataset phase {phase} is not recognized.")

    return train_loader, train_sampler, val_loader, total_epochs, total_iters


def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=True)

    torch.backends.cudnn.benchmark = True  # allowing cuDNN to select the fastest algorithms.
    # torch.backends.cudnn.deterministic = True # insure reproducibility by forcing cuDNN to use deterministic algorithms, at the cost of performance.

    # automatic resume ..
    state_folder_path = "experiments/{}/training_states/".format(opt["name"])
    import os

    try:
        states = os.listdir(state_folder_path)
    except:
        states = []

    resume_state = None
    if len(states) > 0:
        max_state_file = "{}.state".format(max([int(x[0:-6]) for x in states]))
        resume_state = os.path.join(state_folder_path, max_state_file)
        opt["path"]["resume_state"] = resume_state

    # load resume states if necessary
    if opt["path"].get("resume_state"):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
    else:
        resume_state = None

    # mkdir for experiments and logger
    # tb:TensorBoard
    if resume_state is None:
        make_exp_dirs(opt)
        if opt["logger"].get("use_tb_logger") and "debug" not in opt["name"] and opt["rank"] == 0:
            mkdir_and_rename(osp.join("tb_logger", opt["name"]))

    # initialize loggers
    logger, tb_logger = init_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result

    # create model
    # model_type: ImageCleanModel
    if resume_state:  # resume training
        check_resume(opt, resume_state["iter"])
        model = create_model(opt)
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch = resume_state["epoch"]
        current_iter = resume_state["iter"]
    else:
        model = create_model(opt)
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt["datasets"]["train"].get("prefetch_mode")
    if prefetch_mode is None or prefetch_mode == "cpu":
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == "cuda":
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f"Use {prefetch_mode} prefetch dataloader")
        if opt["datasets"]["train"].get("pin_memory") is not True:
            raise ValueError("Please set pin_memory=True for CUDAPrefetcher.")
    else:
        raise ValueError(f"Wrong prefetch_mode {prefetch_mode}.Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(f"Start training from epoch: {start_epoch}, iter: {current_iter}")
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    # for epoch in range(start_epoch, total_epochs + 1):

    iters = opt["datasets"]["train"].get("iters")
    batch_size = opt["datasets"]["train"].get("batch_size_per_gpu")
    # batch_size_per_gpu:控制迭代总epoch，每次总会抽出固定的样本数量，若当前mini_batch_per_gpu小于该数字，则舍弃部分样本，但还是取出固定数量的
    mini_batch_sizes = opt["datasets"]["train"].get("mini_batch_sizes")  # mini_batch_per_gpu：当前patch_size的batch数量
    gt_size = opt["datasets"]["train"].get("gt_size")
    mini_gt_sizes = opt["datasets"]["train"].get("gt_sizes")

    groups = np.array([sum(iters[0 : i + 1]) for i in range(0, len(iters))])
    # 计算iter数组前i项的总数,[92000,64000,48000,36000,36000,24000]既是[92000,156000,204000,240000,276000,300000]

    logger_j = [True] * len(groups)  # [True,True,True,True,True,Tre]

    scale = opt["scale"]

    epoch = start_epoch
    while current_iter <= total_iters:
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()  # 读取数据

        while train_data is not None:
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(current_iter, warmup_iter=opt["train"].get("warmup_iter", -1))

            ### ------Progressive learning ---------------------
            j = ((current_iter > groups) != True).nonzero()[0]
            # j tells you which progressive learning stage you are currently in, or more precisely, which stages you haven't yet completed.
            # current_iter > groups 是一个布尔数组,如[True,True,False,False,False,False],加上!=True,就变成了[False,False,True,True,True,True]
            # nonzero():返回非零元素的索引（对于布尔数组，非零表示 True),即返回布尔数组中 True 值的索引,[0] means the first array,如[2,3,4,5]
            if len(j) == 0:
                bs_j = (
                    len(groups) - 1
                )  # bs_j代表着对应progressive learning的batch_size的索引;此时即是训练的最后一个阶段
            else:
                bs_j = j[0]  # 否则取值为当前阶段的索引，如2

            # 所以bs_j变量的作用是提供正确的索引，以确定size和batch
            mini_gt_size = mini_gt_sizes[bs_j]  # 当前progressive learning阶段的mini_gt_sizess
            mini_batch_size = mini_batch_sizes[bs_j]  # 当前progressive learning阶段的mini_batch_sizes

            if logger_j[bs_j]:
                logger.info(
                    "\n Updating Patch_Size to {} and Batch_Size to {} \n".format(
                        mini_gt_size, mini_batch_size * torch.cuda.device_count()
                    )
                )
                logger_j[bs_j] = False  # 每个阶段只输出一次该信息

            lq = train_data["lq"]  # low quality
            gt = train_data["gt"]

            if (
                mini_batch_size < batch_size
            ):  # 如果当前progressive learning的mini_batch_size比batch_size_per_gpu小，那么就按照小的实现。
                indices = random.sample(range(0, batch_size), k=mini_batch_size)
                lq = lq[indices]
                gt = gt[indices]

            if mini_gt_size < gt_size:  # 如果当前的progressive learning的mini_gt_size小于gt_size,就裁剪
                x0 = int((gt_size - mini_gt_size) * random.random())
                y0 = int((gt_size - mini_gt_size) * random.random())
                x1 = x0 + mini_gt_size
                y1 = y0 + mini_gt_size
                lq = lq[:, :, x0:x1, y0:y1]
                gt = gt[:, :, x0 * scale : x1 * scale, y0 * scale : y1 * scale]
            ###-------------------------------------------

            model.feed_train_data({"lq": lq, "gt": gt})
            model.optimize_parameters(current_iter)

            iter_time = time.time() - iter_time
            # log
            if current_iter % opt["logger"]["print_freq"] == 0:
                log_vars = {"epoch": epoch, "iter": current_iter}
                log_vars.update({"lrs": model.get_current_learning_rate()})
                log_vars.update({"time": iter_time, "data_time": data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # save models and training states
            if current_iter % opt["logger"]["save_checkpoint_freq"] == 0:
                logger.info("Saving models and training states.")
                model.save(epoch, current_iter)

            # validation
            if opt.get("val") is not None and (current_iter % opt["val"]["val_freq"] == 0):
                rgb2bgr = opt["val"].get("rgb2bgr", True)
                # wheather use uint8 image to compute metrics
                use_image = opt["val"].get("use_image", True)
                model.validation(
                    val_loader,
                    current_iter,
                    tb_logger,
                    opt["val"]["save_img"],
                    rgb2bgr,
                    use_image,
                )

            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()
        # end of iter
        epoch += 1

    # end of epoch

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f"End of training. Time consumed: {consumed_time}")
    logger.info("Save the latest model.")
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get("val") is not None:
        model.validation(val_loader, current_iter, tb_logger, opt["val"]["save_img"])
    if tb_logger:
        tb_logger.close()


if __name__ == "__main__":
    main()
