import math
import torch
from models.quaternion_distances import quaternion_distance
from torch.utils.tensorboard import SummaryWriter
import time
import os
import csv

def count_error(target_transl, target_rot, transl_err, rot_err):
    total_trasl_error = torch.tensor(0.0).cuda()
    total_rot_error = quaternion_distance(target_rot, rot_err,
                                          target_rot.device)
    total_rot_error = total_rot_error * 180. / math.pi
    for j in range(target_transl.shape[0]):
        total_trasl_error += torch.norm(target_transl[j] -
                                        transl_err[j]) * 100.
    trasl_e = total_trasl_error.item()
    rot_e = total_rot_error.sum().item()
    return trasl_e, rot_e


class Logger:

    def __init__(self, args, scheduler, loader_size, split, models_num=1):
        ### ========================================
        #   --->>  终端打印训练数据/tensorboard <<-----
        ### ====================================
        self.split = split
        self.total_steps = 0
        self.compensation_step = 0  # 让每一轮最后不满频率数据 清零

        self.total_start_time = time.time()
        self.epoch_start_time = time.time()
        self.start_time = time.time()  # 5 * log_frequency 的记录时间

        self.log_frequency = args.log_frequency  # 记录 tensorboard 频率
        self.print_frequency = args.log_frequency  # 打印终端频率
        self.dataloader_size = loader_size  #打印
        self.scheduler = scheduler
        self.local_loss = 0.  # 只是一定 log_frequency 下，输出loss均值
        self.total_loss = 0.  # 每 epoch 输出loss均值

        self.total_val_t = 0.
        self.total_val_r = 0.

        self.save_path = self.__init_save_path(args, models_num)  #返回路径词典
        self.writer = self.__init_log_tensorboard(args, split)
 
    def _print_fre_status(self):

        iter = f'Iter {(self.total_steps ) % self.dataloader_size}/{self.dataloader_size}, '
        mean_local_loss = f'{self.split}-loss = {self.local_loss/self.print_frequency:.3f}, '
        during_time = f'time for {self.print_frequency} iter: {time.time()-self.start_time:.4f}'
        self.start_time = time.time()
        print(iter + mean_local_loss + during_time)

    def push_loss(self, loss, trasl_e=None, rot_e=None):
        self.total_steps += 1
        self.local_loss += loss['total_loss'].item()
        self.total_loss += loss['total_loss'].item()
        if trasl_e != None and rot_e != None:
            self.total_val_t += trasl_e
            self.total_val_r += rot_e

        # 状态-记录至 tensorboard
        if self.total_steps % self.log_frequency == 0:
            self.write_dict(loss)

        # 状态-打印至终端
        if (self.total_steps -
                self.compensation_step) % self.print_frequency == 0:
            self._print_fre_status()
            self.local_loss = 0

    def print_epoch_status(self, epoch, dataset_size):
        print("------------------------------------")
        print('epoch %d total loss = %.3f' %
              (epoch, (self.total_loss / dataset_size)*100))
        print('Total epoch time = %.2f' %
              (time.time() - self.epoch_start_time))
        if self.scheduler is not None:
            self.writer.add_scalar("LR",
                                   self.scheduler.get_last_lr()[0], epoch)

        self.start_time = time.time()
        self.epoch_start_time = time.time()
        self.local_loss = 0
        self.total_loss = 0
        self.compensation_step = self.total_steps % self.print_frequency

        # 验证集计算平移选旋转误差
        if self.total_val_r != 0 or self.total_val_t != 0:
            print(
                f'{self.split}-total traslation error: {self.total_val_t / dataset_size} cm'
            )
            print(
                f'{self.split}-total rotation error: {self.total_val_r / dataset_size} °'
            )
            self.writer.add_scalar("Val_t_error",
                                   self.total_val_t / dataset_size, epoch)
            self.writer.add_scalar("Val_r_error",
                                   self.total_val_r / dataset_size, epoch)
            self.total_val_t = 0.
            self.total_val_r = 0.
        print("------------------------------------")

    def write_dict(self, results):
        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        print('full training time = %.2f HR' %
              ((time.time() - self.start_full_time) / 3600))
        self.writer.close()

    def __init_save_path(self, args, models_num):
        args.output = os.path.join(args.output, args.iterative_method)
        save_path = {}
        ### ============================
        #   --->>  保存 model 路径 <<-----
        ### ============================
        model_savepath = os.path.join(args.checkpoints,
                                      'val_seq_' + args.val_sequence, 'models')
        if not os.path.exists(model_savepath):
            os.makedirs(model_savepath)
        save_path['model_savepath'] = model_savepath
        ### ============================
        #   --->>  保存图片输出路径 <<-----
        ### ============================
        rgb_path = os.path.join(args.output, 'rgb')
        if not os.path.exists(rgb_path):
            os.makedirs(rgb_path)
        save_path['rgb_path'] = rgb_path

        depth_path = os.path.join(args.output, 'depth')
        if not os.path.exists(depth_path):
            os.makedirs(depth_path)
        save_path['depth_path'] = depth_path

        input_path = os.path.join(args.output, 'input')
        if not os.path.exists(input_path):
            os.makedirs(input_path)
        save_path['input_path'] = input_path

        gt_path = os.path.join(args.output, 'gt')
        if not os.path.exists(gt_path):
            os.makedirs(gt_path)
        save_path['gt_path'] = gt_path

        if args.out_fig_lg == 'EN':
            results_path = os.path.join(args.output, 'results_en')
        elif args.out_fig_lg == 'CN':
            results_path = os.path.join(args.output, 'results_cn')
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        save_path['results_path'] = results_path

        pred_path = os.path.join(args.output, 'pred')
        for it in range(models_num):
            pred_path_it = os.path.join(pred_path, 'iteration_' + str(it + 1))
            if not os.path.exists(pred_path_it):
                os.makedirs(pred_path_it)
            save_path['pred_path_it'] = pred_path_it

        ### =================================
        #   ---->>  保存 pcd点云 输出路径 <<----
        ### =================================
        pc_lidar_path = os.path.join(args.output, 'pointcloud', 'lidar')
        if not os.path.exists(pc_lidar_path):
            os.makedirs(pc_lidar_path)
        save_path['pc_lidar_path'] = pc_lidar_path

        pc_input_path = os.path.join(args.output, 'pointcloud', 'input')
        if not os.path.exists(pc_input_path):
            os.makedirs(pc_input_path)
        save_path['pc_input_path'] = pc_input_path

        pc_pred_path = os.path.join(args.output, 'pointcloud', 'pred')
        if not os.path.exists(pc_pred_path):
            os.makedirs(pc_pred_path)
        save_path['pc_pred_path'] = pc_pred_path

        return save_path

    def __init_log_tensorboard(self, args, split):
        log_tensorboard_savepath = os.path.join(args.checkpoints,
                                                'val_seq_' + args.val_sequence,
                                                'models')
        if not os.path.exists(log_tensorboard_savepath):
            os.makedirs(log_tensorboard_savepath)
        writer = SummaryWriter(os.path.join(log_tensorboard_savepath, split))
        return writer