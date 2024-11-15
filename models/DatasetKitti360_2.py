# 随机设置一定范围内的标定参数扰动值
# train的时候每次都随机生成 T-init ,每个epoch使用不同的参数
# test则在初始化的时候提前设置好,每个epoch都使用相同的参数
# ------------------------#
import sys

# sys.path.append('/home/bolo/models/HliuNet')


import csv
import os
from math import radians

import mathutils
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TTF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from models.utils import invert_pose, rotate_forward, quaternion_from_matrix, merge_inputs, get_decalib, lidar_project_depth, rotate_back, show_tensor_imgs
# from utils.utils import invert_pose, rotate_forward, quaternion_from_matrix, merge_inputs, get_decalib, lidar_project_depth, rotate_back, show_tensor_imgs
from pykitti import odometry

import torch.utils.data as data
import random
import torch.nn.functional as F
from kitti360scripts.devkits.commons import loadCalibration

kitti360_sequence_dict = {
    "00": "2013_05_28_drive_0000_sync", 
    # "02": "2013_05_28_drive_0002_sync", 
    "03": "2013_05_28_drive_0003_sync", 
    "04": "2013_05_28_drive_0004_sync", 
    "05": "2013_05_28_drive_0005_sync", 
    "06": "2013_05_28_drive_0006_sync", 
    # "07": "2013_05_28_drive_0007_sync", 
    "09": "2013_05_28_drive_0009_sync", 
    # "10": "2013_05_28_drive_0010_sync"
}

class DatasetKitti360(Dataset):
    """
    self.all_files = [] 获取所有文件列表 
    self.K = {} K内参 
    self.GTs_T_cam02_velo = {} 外参真值
    self.val_RT = [] T-init 读取/生成
    self.use_reflectance  使用反射率（强度信息）

    self.transform 未使用
    self.augmentation 未使用
    """
    """
    args.data_folder,
    max_r=args.max_r,
    max_t=args.max_t,
    split=split,
    use_reflectance=args.use_reflectance,
    val_sequence=args.val_sequence)
    """

    def __init__(self, args, split='val', device='cpu', suf='.png'):
        super(DatasetKitti360, self).__init__()

        self.use_reflectance = args.use_reflectance
        self.maps_folder = ''
        self.device = device
        self.max_r = args.max_r
        self.max_t = args.max_t
        #.....
        self.max_depth = args.max_depth
        self.img_shape = (384, 1280) #args.img_shape
        self.use_reflectance = args.use_reflectance
        # self.augmentation = args.augmentation
        self.root_dir = args.data_folder
        # self.transform = args.transform
        self.split = split
        self.val_sequence ='03'# args.val_sequence
        self.CTs_T_cam00_to_velo = {}
        self.K = {}
        self.suf = suf
        self.all_files = []
        self.val_RT = []

        self.sequence_list = [
        #    "00"
           "03", "04"
        #    , "05", "06","09"
        ]  # '00',# 00 is test
        print("max_t", self.max_t,self.max_r)
        #获取 内参、外参真值和all_files
        for seq in self.sequence_list:
            filePersIntrinsic = os.path.join(args.data_folder, 'calibration', 'perspective.txt')
            fileCameraToVelo = os.path.join(args.data_folder, 'calibration', 'calib_cam_to_velo.txt')
            # 相机内参字典 P_rect_00 and P_rect_01 
            calib = loadCalibration.loadPerspectiveIntrinsic(filePersIntrinsic) # 4x4
            # 左透视摄像机图像 00 坐标到 Velodyne 坐标的刚性变换。
            T_cam00_velo_np = loadCalibration.loadCalibrationRigid(fileCameraToVelo)
            self.K[seq] = calib["P_rect_00"][: 3, : 3] # 3x3
            self.CTs_T_cam00_to_velo[seq] = np.linalg.inv(T_cam00_velo_np) # T_cam00_velo_np #4x4
            # print("self.K[seq]")
            # print(self.K[seq])
            # print("self.CTs_T_cam00_to_velo[seq]")
            # print(self.CTs_T_cam00_to_velo[seq])
            print("sep===",seq)
            image_list = os.listdir(
                os.path.join(args.data_folder, "data_2d_raw", kitti360_sequence_dict[seq], "image_00", "data_rect"))
            image_list.sort()
            if len(image_list) >= 4000:
                image_list = image_list[0:4000:1]
                # print(image_list)
        
            # 获取指定目录下的图像文件列表并按名称排序
            for image_name in image_list:
                if not os.path.exists(
                    os.path.join(args.data_folder, "data_2d_raw", kitti360_sequence_dict[seq], "velodyne_points", "data", 
                                 str(image_name.split('.')[0]) + '.bin')):
                    print("No exist!")
                    continue
                if not os.path.exists(
                    os.path.join(args.data_folder, "data_2d_raw", kitti360_sequence_dict[seq], "image_00", "data_rect", 
                                 str(image_name.split('.')[0]) + suf)):
                    print("No exist!")
                    continue

                # if seq == args.val_sequence:
                #     if split.startswith('val') or split =="test":
                #         self.all_files.append(os.path.join(seq,image_name.split('.')[0]))
                # elif (not seq == args.val_sequence) and split == 'train':
                #     self.all_files.append(os.path.join(seq, image_name.split('.')[0]))



                if seq == self.val_sequence and self.split == 'val':
                    
                    self.all_files.append(os.path.join(seq,image_name.split('.')[0]))
                elif (not seq == self.val_sequence) and self.split == 'train':
                    
                    self.all_files.append(os.path.join(seq, image_name.split('.')[0]))     
                
            print('ZGHZall_files', len(self.all_files))

        # T-init 读取/生成
        # self.val_RT = []
        if split == 'val' or split == 'test':
            val_RT_file = os.path.join(
                self.root_dir,
                f'kitti_360_val_RT_left_seq{self.val_sequence}_{self.max_r:.2f}_{self.max_t:.2f}.csv'
            )
            if os.path.exists(val_RT_file):
                print(f'VAL SET: Using this file: {val_RT_file}')
                df_test_RT = pd.read_csv(val_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.val_RT.append(list(row))
            else:
                print(f'VAL SET - Not found: {val_RT_file}')
                print("Generating a new one")
                val_RT_file = open(val_RT_file, 'w')
                val_RT_file = csv.writer(val_RT_file, delimiter=',')
                val_RT_file.writerow(
                    ['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(len(self.all_files)):
                    transl_x, transl_y, transl_z, rotx, roty, rotz = get_decalib(
                        self.max_r, self.max_t)
                    val_RT_file.writerow(
                        [i, transl_x, transl_y, transl_z, rotx, roty, rotz])
                    self.val_RT.append([
                        float(i),
                        float(transl_x),
                        float(transl_y),
                        float(transl_z),
                        float(rotx),
                        float(roty),
                        float(rotz)
                    ])
                print('len val_RT', len(self.val_RT))
            assert len(self.val_RT) == len(
                self.all_files), "Something wrong with test RTs"

    def custom_transform(self, rgb, img_rotation=0., flip=False):
        """ 
        将图像转换为张量并进行标准化后返回。
        如果self.split等于'train'，则会应用颜色变换、水平翻转和旋转操作。
        """
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        if self.split == 'train':
            color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
            rgb = color_transform(rgb)
            if flip:
                rgb = TTF.hflip(rgb)
            rgb = TTF.rotate(rgb, img_rotation)

        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        item = self.all_files[idx]
        seq = str(item.split('/')[0])
        rgb_name = str(item.split('/')[1])
        
        img_path = os.path.join(self.root_dir, "data_2d_raw", kitti360_sequence_dict[seq], "image_00", "data_rect", 
                                rgb_name + self.suf)
        lidar_path = os.path.join(self.root_dir, "data_2d_raw", kitti360_sequence_dict[seq], "velodyne_points", "data",
                                  rgb_name + '.bin')

        # ===============================================
        ### --------->>>  相机数据处理 <<<-----------------
        # ===============================================
        img = Image.open(img_path)
        img_rotation = 0.
        h_mirror = False
        try:
            img = self.custom_transform(img, img_rotation, h_mirror)
        except OSError:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)
        real_shape = [img.shape[1], img.shape[2], img.shape[0]]

        # ===============================================
        ### --------->>> 获取 decalib <<<-----------------
        # ===============================================
        # 随机设置一定范围内的标定参数扰动值
        # train的时候每次都随机生成,每个epoch使用不同的参数
        # test则在初始化的时候提前设置好,每个epoch都使用相同的参数
        # Pc` = RT * Pc H_init = H_decalib * H_gt
        if self.split == 'train':
            transl_x, transl_y, transl_z, rotx, roty, rotz = get_decalib(
                self.max_r, self.max_t)
        else:
            initial_RT = self.val_RT[idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]
        euler_R = mathutils.Euler((rotx, roty, rotz))
        vector_T = mathutils.Vector((transl_x, transl_y, transl_z))

        quaternion_R, vector_T = invert_pose(euler_R, vector_T)
        tensor_q_R, tensor_v_T = torch.tensor(quaternion_R), torch.tensor(
            vector_T)
        matrix_R = mathutils.Quaternion(tensor_q_R).to_matrix()
        matrix_R.resize_4x4()
        matrix_T = mathutils.Matrix.Translation(tensor_v_T)
        RT_decalib = matrix_T * matrix_R

        # ===============================================
        ### --------->>> Lidar数据处理 <<<-----------------
        # ===============================================
        lidar_scan = np.fromfile(lidar_path, dtype=np.float32)
        # 每行有4个元素。这样做是因为LIDAR扫描通常会生成(x, y, z, intensity)四个值作为一个点的信息。
        pc = lidar_scan.reshape((-1, 4))

        # 通过筛选条件创建一个布尔数组_indices，用于标记在x和y轴上超出范围[-3, 3]的点云。
        valid_indices = pc[:, 0] < -3.
        valid_indices = valid_indices | (pc[:, 0] > 3.)
        if self.max_depth < 80.:
            valid_indices = valid_indices | pc[0, :] < self.max_depth
        valid_indices = valid_indices | (pc[:, 1] < -3.)
        valid_indices = valid_indices | (pc[:, 1] > 3.)

        # 点云shape N * 4
        pc = pc[valid_indices].copy()
        reflectance = None
        pc_org = torch.from_numpy(pc.astype(np.float32))

        # 是否使用强度信息 N ,没有1*N 或者 N*1
        if self.use_reflectance:
            reflectance = pc[:, 3].copy()
            reflectance = torch.from_numpy(reflectance).float()

        #确认点云形状，并将 x,y,z,1
        # 如果是，说明点云数据已经是所需的形状，不需要进行转换。 4 * N
        if pc_org.shape[1] == 4 or pc_org.shape[1] == 3:
            pc_org = pc_org.t()
        if pc_org.shape[0] == 3:
            homogeneous = torch.ones(pc_org.shape[1]).unsqueeze(0)
            pc_org = torch.cat((pc_org, homogeneous), 0)
        elif pc_org.shape[0] == 4:
            if not torch.all(pc_org[3, :] == 1.):
                pc_org[3, :] = 1.
        else:
            raise TypeError("Wrong PointCloud shape")

        ## -----------------------------------------------------------------
        #向量在基A下坐标为α，求它在基B下的坐标β。 -- RT numpy 类型
        # A→B的 变换矩阵  为P。β=(P^-1)α。
        RT_gt = self.CTs_T_cam00_to_velo[seq].astype(np.float32)
        pc_rot_gt = np.matmul(RT_gt, pc_org.numpy())
        pc_rot_gt = pc_rot_gt.astype(np.float32).copy()
        pc_gt = torch.from_numpy(pc_rot_gt)
        # pc_gt.cuda()

        calib = self.K[seq]
        pc_lidar = pc_gt.clone()

        depth_gt, uv, reflect_img_gt = lidar_project_depth(pc_lidar,
                                                           calib,
                                                           real_shape,
                                                           reflectance=None)
        depth_gt /= self.max_depth

        # Pc` = RT * Pc // H_init = H_decalib * H_gt
        # print(pc_lidar == pc_gt) # True
        pc_rotated = rotate_back(pc_lidar, RT_decalib)
        depth_img, uv, reflect_img = lidar_project_depth(
            pc_rotated, calib, real_shape, reflectance)
        depth_img /= self.max_depth
        # show_tensor_imgs(depth_gt, reflect_img_gt, depth_img, reflect_img)
        if self.use_reflectance:
            lidar_input = torch.cat([depth_img, reflect_img], dim=0)
        else:
            lidar_input = depth_img

        # ==================================================
        # --------------->>> 填充大小一致 <<<-----------------
        # ==================================================
        # odom 数据集图像大小不一致
        # ProjectPointCloud in RT-pose rgb--torch.Size([3, 376, 1241])
        shape_pad = [0, 0, 0, 0]
        # img_shape = (384, 1280)  rgb--torch.Size([3, 376, 1241])
        shape_pad[3] = (self.img_shape[0] - img.shape[1])  # // 2
        shape_pad[1] = (self.img_shape[1] - img.shape[2])  # // 2 + 1
        img = F.pad(img, shape_pad)
        lidar_input = F.pad(lidar_input, shape_pad)
        depth_gt = F.pad(depth_gt, shape_pad)
        """
        'rgb': img,  # 归一化/torch 图像， 本实例未镜像
        'point_cloud': pc_in,  # camera 坐标系下，X轴与Y轴过滤后的点云真值
        'calib': calib,  # 相机内参 self.K[seq]
        'tr_error': T,  # decalib 误差
        'rot_error': R,
        'seq': int(seq),  #以上一样
        'img_path': img_path,
        'rgb_name': rgb_name + '.png',
        'item': item,
        'extrin': RT,  # 真值
        'initial_RT': initial_RT
        """
        sample = {
            'rgb': img,  # 归一化/torch 图像， 本实例未镜像
            'lidar_input': lidar_input,
            'depth_gt': depth_gt,
            'point_cloud': pc_gt,  # camera 坐标系下，X轴与Y轴过滤后的点云真值
            'calib': calib,  # 相机内参 self.K[seq]
            'real_shape': real_shape,  # 配合内参，depth_img_pre
            'tr_error': tensor_v_T,  # decalib 误差
            'rot_error': tensor_q_R,
            'extrin': RT_gt,  # 真值
        }
        # if self.split == 'test':
        #     if not self.use_reflectance:
        #         sample = {
        #             'rgb': img,  # 归一化/torch 图像， 本实例未镜像
        #             'point_cloud': pc_in,  # camera 坐标系下，X轴与Y轴过滤后的点云真值
        #             'calib': calib,  # 相机内参 self.K[seq]
        #             'tr_error': T,  # decalib 误差
        #             'rot_error': R,
        #             # 'seq': int(seq),  #以上一样
        #             'img_path': img_path,
        #             'rgb_name': rgb_name + '.png',
        #             'item': item,
        #             'extrin': RT,  # 真值
        #             'initial_RT': initial_RT
        #         }  # self.val_RT[idx] 误差
        #     else:
        #         sample = {
        #             'reflectance': reflectance,
        #             'rgb': img,  # 归一化/torch 图像， 本实例未镜像
        #             'point_cloud': pc_in,  # camera 坐标系下，X轴与Y轴过滤后的点云真值
        #             'calib': calib,  # 相机内参 self.K[seq]
        #             'tr_error': T,  # decalib 误差
        #             'rot_error': R,
        #             # 'seq': int(seq),  #以上一样
        #             'img_path': img_path,
        #             'rgb_name': rgb_name + '.png',
        #             'item': item,
        #             'extrin': RT,  # 真值
        #             'initial_RT': initial_RT
        #         }  # self.val_RT[idx] 误差
        # else:
        #     if not self.use_reflectance:
        #         sample = {
        #             'rgb': img,
        #             'point_cloud': pc_in,
        #             'calib': calib,
        #             'tr_error': T,
        #             'rot_error': R,
        #             # 'seq': int(seq),
        #             'rgb_name': rgb_name,
        #             'item': item,
        #             'extrin': RT
        #         }
        #     else:
        #         sample = {
        #             'reflectance': reflectance,
        #             'rgb': img,
        #             'point_cloud': pc_in,
        #             'calib': calib,
        #             'tr_error': T,
        #             'rot_error': R,
        #             # 'seq': int(seq),
        #             'rgb_name': rgb_name,
        #             'item': item,
        #             'extrin': RT
        #         }

        return sample

EPOCH = 1


def _init_fn(worker_id, seed):
    """
    在多进程数据加载情况下，
    主函数中的随机种子和 worker_init_fn 中的随机种子会相互影响，
    可能会导致实验的不可复现性。为了避免这种情况，
    你应该在 worker_init_fn 中设置每个 worker 的种子，而在主函数中不要再额外设置种子。
    """
    global EPOCH
    seed = seed + worker_id + EPOCH * 100
    print(f"Init worker {worker_id} with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    EPOCH += 1


def init_fn(x):
    seed = torch.initial_seed() % 2**32
    return _init_fn(x, seed)


def fetch_dataloader(args, split='train'):
    if args.stage == 'kitti':
        print("This kitti-odometry dataset!")
        dataset_class = DatasetKitti360
        dataset_kitti = dataset_class(args, split=split)
        dataset_kitti_size = len(dataset_kitti)
        print('Number of the {} Dataset: {}'.format(split, dataset_kitti_size))

    if split == 'train':
        shuffle = True
    else:
        shuffle = False

    dataloader_kitti = data.DataLoader(
        dataset=dataset_kitti,
        shuffle=shuffle,
        batch_size=args.batch_size,
        num_workers=args.num_worker,
        worker_init_fn=init_fn,
        collate_fn=merge_inputs,
        drop_last=False,
        #    pin_memory=True
    )
    dataloader_kitti_size = len(dataloader_kitti)
    print('Number of the {} DataLoader: {}'.format(split,
                                                   dataloader_kitti_size))

    return dataloader_kitti, dataset_kitti_size


if __name__ == '__main__':
    import sys

    sys.path.append('/home/x-go/Desktop/HliuNet_0920_kitti_360')
    import setparams
    args = setparams.setparam()

    print("starting")
    dataset_class = DatasetKitti360
    dataset_train = dataset_class(args, split='train')
    for i in range(20):
        data = dataset_train[i]
        print(data)
    print(1)
    print(2)
