from __future__ import print_function, division  # 使得print语句变为函数形式，并且使整数除法返回浮点数结果
import sys
sys.path.append('.')
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models.ZghNet import ZghNet  # Hliu
from models.utils import merge_inputs
from models.logger_01 import Logger, count_error
from models.DatasetLidarCamera import fetch_dataloader

import torch.utils.data as data
from models.losses import CombinedLoss

EPOCH = 1

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6 #混合精度加速
    class GradScaler:

        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model, train_loader=1):
    """  创建优化器和学习率调度器  """
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.Adam(parameters,
                           lr=args.BASE_LEARNING_RATE,
                           weight_decay=5e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=((args.epochs - args.starting_epoch) * train_loader))
    return optimizer, scheduler


def train(args):
    global EPOCH
    model = ZghNet(args)
    print("Model:ZGH", model)
    print("Parameter Count: %d" % count_parameters(model))

    starting_epoch = args.starting_epoch
    model = nn.DataParallel(model)#, device_ids=[3])

    if args.weight is not None:  #加载模型
        print(f"Loading weights from {args.weight}")
        model.load_state_dict(torch.load(args.weight, map_location='cpu'),
                              strict=True)

    
    model.cuda()
    model.train()

    train_loader, train_set_size = fetch_dataloader(args=args, split="train")
    val_loader, val_set_size = fetch_dataloader(args=args, split="val")

    optimizer, scheduler = fetch_optimizer(args, model, len(train_loader))
    scaler = GradScaler(enabled=args.mixed_precision)
    loss_fn = CombinedLoss(args.rescale_transl, args.rescale_rot,
                           args.weight_point_cloud)
    logger_train = Logger(args, scheduler, len(train_loader), 'train')  # Hliu
    logger_val = Logger(args, scheduler, len(val_loader), 'val')

    # 若当前最优移除旧文件，记录最优样本权重
    BEST_VAL_LOSS = 10000.
    old_save_filename = None

    for epoch in range(starting_epoch, args.epochs + 1):
        print('This is %d-th epoch' % epoch)
        EPOCH = epoch

        for batch_idx, sample in enumerate(train_loader):
            optimizer.zero_grad()
            # 将数据放入 GPU ，注意 sample['point_cloud'] 是 list 类型
            sample['tr_error'] = sample['tr_error'].cuda()
            sample['rot_error'] = sample['rot_error'].cuda()
            sample['point_cloud'] = [tensor.cuda() for tensor in sample['point_cloud']]

            rgb_input = sample['rgb'].cuda()
            lidar_input = sample['lidar_input'].cuda()


            rgb_input = F.interpolate(rgb_input,
                                      size=args.img_shape,
                                      mode="bilinear",
                                      align_corners=False)
            lidar_input = F.interpolate(lidar_input,
                                        size=args.img_shape,
                                        mode="bilinear",
                                        align_corners=False)

            # Run model
            T_predicted, R_predicted = model(rgb_input, lidar_input)
            loss = loss_fn(sample['point_cloud'], sample['tr_error'],
                           sample['rot_error'], T_predicted, R_predicted)

            # 这段代码是用于检查损失函数中是否存在NaN值。它
            for key in loss.keys():
                if loss[key].item() != loss[key].item():
                    raise ValueError("Loss {} is NaN".format(key))
            """
            但在处理每一个batch时并不需要与其他batch的梯度混合起来累积计算，
            因此需要对每个batch调用一遍zero_grad（）
            """
            total_loss = loss['total_loss']
            # 调整学习率
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            # 记录结果
            logger_train.push_loss(loss)

        logger_train.print_epoch_status(epoch=epoch,
                                        dataset_size=train_set_size)
        for batch_idx, sample in enumerate(val_loader):
            # 将数据放入 GPU ，注意 sample['point_cloud'] 是 list 类型
            sample['tr_error'] = sample['tr_error'].cuda()
            sample['rot_error'] = sample['rot_error'].cuda()
            sample['point_cloud'] = [
                tensor.cuda() for tensor in sample['point_cloud']
            ]
            rgb_input = sample['rgb'].cuda()
            lidar_input = sample['lidar_input'].cuda()

            # tensorboard 记录 点云投影到图像图需要的数据
            # rgb_show = rgb_input[0].clone()
            # # 选择第一个元素，并裁剪成 (1, H, W) 形状
            # lidar_show = lidar_input[0][0:1, :, :].clone()

            rgb_input = F.interpolate(rgb_input,
                                      size=args.img_shape,
                                      mode="bilinear",
                                      align_corners=False)
            lidar_input = F.interpolate(lidar_input,
                                        size=args.img_shape,
                                        mode="bilinear",
                                        align_corners=False)

            # Run model
            model.eval()
            with torch.no_grad():
                T_predicted, R_predicted = model(rgb_input, lidar_input)
            loss = loss_fn(sample['point_cloud'], sample['tr_error'],
                           sample['rot_error'], T_predicted, R_predicted)
            trasl_e, rot_e = count_error(sample['tr_error'],
                                         sample['rot_error'], T_predicted,
                                         R_predicted)
            # 这段代码是用于检查损失函数中是否存在NaN值。它
            for key in loss.keys():
                if loss[key].item() != loss[key].item():
                    raise ValueError("Loss {} is NaN".format(key))

            logger_val.push_loss(loss, trasl_e, rot_e)

        # 若当前模型参数最优，记录当前模型权重
        val_loss = (logger_val.total_loss / val_set_size) * 100
        if val_loss < BEST_VAL_LOSS:
            BEST_VAL_LOSS = val_loss
            savefilename = f'{logger_train.save_path["model_savepath"]}/checkpoint_r{args.max_r:.2f}_t{args.max_t:.2f}_e{epoch}_val{val_loss:.3f}.pth'
            torch.save(model.state_dict(), savefilename)
            print(f'Model saved as {savefilename}')
            if old_save_filename is not None:
                if os.path.exists(old_save_filename):
                    os.remove(old_save_filename)
            old_save_filename = savefilename

        logger_val.print_epoch_status(epoch=epoch, dataset_size=val_set_size)
    print('full training time = %.2f HR' %
          ((time.time() - logger_train.total_start_time) / 3600))

    return savefilename


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ===============================================
    ### --------->>>  dataloader 实例化 <<<-----------
    # ===============================================
    # dataloader
    parser.add_argument('--stage', default='kitti', help="")
    parser.add_argument('--batch_size', type=int, default=6)  #12 炸
    parser.add_argument('--num_worker', type=int, default=1)
    # dataset 实例化参数
    parser.add_argument(
        '--data_folder',
        default='/data_home/home/zhangguanghui/dataset/',
        help="")
    parser.add_argument('--max_r', default=20., help="")  #1.
    parser.add_argument('--max_t', default=1.5, help="")  #0.1
    parser.add_argument('--use_reflectance', action='store_true', default=True)
    parser.add_argument('--val_sequence', type=str, default='04', help="")
    parser.add_argument('--img_shape', default=[256,512], help='图像统一大小')
    # ===============================================
    ### --->>>  model 实例化参数 and loss损失函数 <<<---
    # ===============================================
    # model 实例化参数
    parser.add_argument(
        '--weight',
        default=None,
        help="")
    parser.add_argument('--name', default='MZ1', help="权重给个名字")
    parser.add_argument('--dropout', default=0.0, help="")

    # train 中 loss
    parser.add_argument('--rescale_transl', type=float, default=2.0)
    parser.add_argument('--rescale_rot', type=float, default=1.0)
    parser.add_argument('--weight_point_cloud', type=float, default=0.5)
    # 优化器和规划器
    parser.add_argument('--BASE_LEARNING_RATE', type=float,
                        default=2e-4)  # 3e-4
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    # train
    parser.add_argument('--max_depth', type=int, default=80)
    parser.add_argument('--mixed_precision',
                        action='store_true',
                        default=True,
                        help='use mixed precision')
    parser.add_argument('--epochs', type=int, default=150)
    # ===============================================
    ### ----------->>>  log 记录实验 <<<--------------
    # ===============================================
    parser.add_argument('--starting_epoch', type=int, default=0)  #10
    parser.add_argument(
        '--checkpoints',
        default='checkpointsZGH/MZ1',
        help="")
    parser.add_argument('--log_frequency', type=int, default=40)  #10

    ######################
    parser.add_argument('--output', default='./output/MZ1', help="")
    parser.add_argument('--iterative_method', default='single',
                        help='')  #'multi_range',
    parser.add_argument('--out_fig_lg', default='CN', help="[EN, CN]")
    parser.add_argument('--save_log', action='store_true', help='')

    args = parser.parse_args()
    train(args)