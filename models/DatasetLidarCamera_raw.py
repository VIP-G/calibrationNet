# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------
"""
Copyright (C) 2020 米兰比可卡大学, iralab
# 作者：Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# 以创作共用协议发布
# 署名-非商业性-相同方式共享 4.0 国际许可协议。
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------

# Modified Author: Xudong Lv
# 基于 github.com/cattaneod/CMRNet/blob/master/DatasetVisibilityKitti.py

"""

"""
# 随机设置一定范围内的标定参数扰动值
# train的时候每次都随机生成,每个epoch使用不同的参数
# test则在初始化的时候提前设置好,每个epoch都使用相同的参数
"""

import argparse
import csv
import os
from math import radians
import cv2

# import h5py
import mathutils
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TTF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from models.utils import invert_pose, rotate_forward, merge_inputs,quaternion_from_matrix
from pykitti import odometry
import pykitti

import torch.utils.data as data
import random


from kitti360scripts.devkits.commons import loadCalibration
import os
from torch.utils.data import Dataset
import pandas as pd
import csv
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TTF
import torch
from PIL import Image
import mathutils
from models.utils import rotate_forward, invert_pose
from math import radians

kitti360_sequence_dict = {
    "00": "2013_05_28_drive_0000_sync", 
    "02": "2013_05_28_drive_0002_sync", 
    "03": "2013_05_28_drive_0003_sync", 
    "04": "2013_05_28_drive_0004_sync", 
    "05": "2013_05_28_drive_0005_sync", 
    "06": "2013_05_28_drive_0006_sync", 
    "07": "2013_05_28_drive_0007_sync", 
    "09": "2013_05_28_drive_0009_sync", 
    "10": "2013_05_28_drive_0010_sync"
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

    def __init__(self,
                 dataset_dir,
                 transform=None,
                 augmentation=False,
                 use_reflectance=False,
                 max_t=1.5,
                 max_r=20.,
                 split='val',
                 device='cpu',
                 val_sequence='03',
                 test_sequence='00',
                 suf='.png'): 
        super(DatasetKitti360, self).__init__()

        self.use_reflectance = use_reflectance
        self.maps_folder = ''
        self.device = device
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dir = dataset_dir
        self.transform = transform
        self.split = split
        self.test_sequence = test_sequence
        self.GTs_R = {}
        self.GTs_T = {}
        self.CTs_T_cam00_to_velo = {}
        self.K = {}
        self.suf = suf
        self.all_files = []
        self.val_RT = []

        self.sequence_list = [
            "03","04","05","06","09"
        ]  # '00',# 00 is test
        print('testing sequence: ', self.test_sequence)
        for seq in self.sequence_list:
            filePersIntrinsic = os.path.join(dataset_dir, 'calibration', 'perspective.txt')
            fileCameraToVelo = os.path.join(dataset_dir, 'calibration', 'calib_cam_to_velo.txt')
            # 相机内参字典 P_rect_00 and P_rect_01 
            calib = loadCalibration.loadPerspectiveIntrinsic(filePersIntrinsic) # 4x4
            # 左透视摄像机图像 00 坐标到 Velodyne 坐标的刚性变换。
            T_cam00_velo_np = loadCalibration.loadCalibrationRigid(fileCameraToVelo)
            self.K[seq] = calib["P_rect_00"][: 3, : 3] # 3x3
            self.CTs_T_cam00_to_velo[seq] = np.linalg.inv(T_cam00_velo_np) #4x4
            # print("self.K[seq]")
            # print(self.K[seq])
            # print("self.CTs_T_cam00_to_velo[seq]")
            # print(self.CTs_T_cam00_to_velo[seq])

            image_list = os.listdir(
                os.path.join(dataset_dir, "data_2d_raw", kitti360_sequence_dict[seq], "image_00", "data_rect"))
            image_list.sort()
            if len(image_list) >= 4000:
                image_list = image_list[0:4000:1]
            for image_name in image_list:
                if not os.path.exists(
                    os.path.join(dataset_dir, "data_2d_raw", kitti360_sequence_dict[seq], "velodyne_points", "data", 
                                 str(image_name.split('.')[0]) + '.bin')):
                    print("No exist!")
                    continue
                if not os.path.exists(
                    os.path.join(dataset_dir, "data_2d_raw", kitti360_sequence_dict[seq], "image_00", "data_rect", 
                                 str(image_name.split('.')[0]) + suf)):
                    print("No exist!")
                    continue

                if seq == self.test_sequence and split == 'test':
                     self.all_files.append(os.path.join(seq,image_name.split('.')[0]))
            print("zghself.all_files",len(self.all_files))
        
        # T-init 读取/生成
        # self.val_RT = []
        if split == 'val' or split == 'test':
            val_RT_file = os.path.join(
                dataset_dir,
                f'kitti_360_val_RT_left_seq{test_sequence}_{max_r:.2f}_{max_t:.2f}.csv')
            
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
                    rotz = np.random.uniform(-max_r,
                                             max_r) * (3.141592 / 180.0)
                    roty = np.random.uniform(-max_r,
                                             max_r) * (3.141592 / 180.0)
                    rotx = np.random.uniform(-max_r,
                                             max_r) * (3.141592 / 180.0)
                    transl_x = np.random.uniform(-max_t, max_t)
                    transl_y = np.random.uniform(-max_t, max_t)
                    transl_z = np.random.uniform(-max_t, max_t)
                    # transl_z = np.random.uniform(-max_t, min(max_t, 1.))
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

            assert len(self.val_RT) == len(
                self.all_files), "Something wrong with test RTs"

    def __len__(self):
        return len(self.all_files)

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
            #io.imshow(np.array(rgb))
            #io.show()

        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb

    def __getitem__(self, idx):
        item = self.all_files[idx]
        seq = str(item.split('/')[0])
        rgb_name = str(item.split('/')[1])

        img_path = os.path.join(self.root_dir, "data_2d_raw", kitti360_sequence_dict[seq], "image_00", "data_rect", 
                                rgb_name + self.suf)
        lidar_path = os.path.join(self.root_dir, "data_2d_raw", kitti360_sequence_dict[seq], "velodyne_points", "data",
                                  rgb_name + '.bin')
        
        lidar_scan = np.fromfile(lidar_path, dtype=np.float32)
        # 每行有4个元素。这样做是因为LIDAR扫描通常会生成(x, y, z, intensity)四个值作为一个点的信息。
        pc = lidar_scan.reshape((-1, 4))

        # 通过筛选条件创建一个布尔数组_indices，用于标记在x和y轴上超出范围[-3, 3]的点云。
        valid_indices = pc[:, 0] < -3.
        valid_indices = valid_indices | (pc[:, 0] > 3.)
        valid_indices = valid_indices | (pc[:, 1] < -3.)
        valid_indices = valid_indices | (pc[:, 1] > 3.)
        pc = pc[valid_indices].copy()
        # 筛选后的点云转换为torch.Tensor类型
        pc_org = torch.from_numpy(pc.astype(np.float32))
        # 是否使用强度信息
        if self.use_reflectance:
            reflectance = pc[:, 3].copy(
            )  # reflectance = pc[:, 3].copy()：这一行代码从变量pc中选择第4列的数据
            reflectance = torch.from_numpy(reflectance).float()
        # for i in range(reflectance.shape[0]):
        # print(f"第 {i} 个 reflection = {reflectance[i]}")
        # print(f"第 {i} 个 pc[i][3] = {pc[i][3]} ")
        # if not reflectance[i] == pc[i][3]:
        # print("程序有问题")
        # else:
        #     print("强度信息没问题")
        ##----------------------------------------------------------------
        RT = self.CTs_T_cam00_to_velo[seq].astype(np.float32)

        # 如果是，说明点云数据已经是所需的形状，不需要进行转换。
        if pc_org.shape[1] == 4 or pc_org.shape[1] == 3:
            pc_org = pc_org.t()
        # 代码检查点云数据的行数是否为3。如果是，说明点云数据是三维坐标（x、y、z），
        # 需要添加一个齐次坐标（homogeneous coordinate）来表示点的位置。
        if pc_org.shape[0] == 3:
            homogeneous = torch.ones(pc_org.shape[1]).unsqueeze(0)
            pc_org = torch.cat((pc_org, homogeneous), 0)
        elif pc_org.shape[0] == 4:
            # 这行代码检查点云数据的第四行是否全为1。如果不是，说明齐次坐标的值不正确，需要将其设置为1。
            if not torch.all(pc_org[3, :] == 1.):
                pc_org[3, :] = 1.
        else:
            raise TypeError("Wrong PointCloud shape")
        ## -----------------------------------------------------------------
        pc_rot = np.matmul(RT, pc_org.numpy())
        # cam坐标系下 lidar 真值
        pc_rot = pc_rot.astype(np.float32).copy()
        pc_in = torch.from_numpy(pc_rot)

        h_mirror = False
        img = Image.open(img_path)
        img_rotation = 0.
        try:
            img = self.custom_transform(img, img_rotation, h_mirror)
        except OSError:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)
        
        # Rotate PointCloud for img_rotation
        # 和图像一样，应该是想增加扰动？？？？Rotate PointCloud for img_rotation
        if self.split == 'train':
            R = mathutils.Euler((radians(img_rotation), 0, 0), 'XYZ')
            T = mathutils.Vector((0., 0., 0.))
            pc_in = rotate_forward(pc_in, R, T)
        
        if self.split == 'train':
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle,
                                     max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle,
                                     max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle,
                                     max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, self.max_t)
            # transl_z = np.random.uniform(-self.max_t, min(self.max_t, 1.))
        else:
            initial_RT = self.val_RT[idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]

        # 随机设置一定范围内的标定参数扰动值
        # train的时候每次都随机生成,每个epoch使用不同的参数
        # test则在初始化的时候提前设置好,每个epoch都使用相同的参数
        R = mathutils.Euler((rotx, roty, rotz))
        T = mathutils.Vector((transl_x, transl_y, transl_z))
        print("rgb_name",rgb_name)
  
        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)

        calib = self.K[seq]
        if h_mirror:
            calib[2] = (img.shape[2] / 2) * 2 - calib[2]

        if self.split == 'test':
            if not self.use_reflectance:
                sample = {
                    'rgb': img,  # 归一化/torch 图像， 本实例未镜像
                    'point_cloud': pc_in,  # camera 坐标系下，X轴与Y轴过滤后的点云真值
                    'calib': calib,  # 相机内参 self.K[seq]
                    'tr_error': T,  # decalib 误差
                    'rot_error': R,
                    # 'seq': int(seq),  #以上一样
                    'img_path': img_path,
                    'rgb_name': rgb_name + '.png',
                    'item': item,
                    'extrin': RT,  # 真值
                    'initial_RT': initial_RT
                }  # self.val_RT[idx] 误差
            else:
                sample = {
                    'reflectance': reflectance,
                    'rgb': img,  # 归一化/torch 图像， 本实例未镜像
                    'point_cloud': pc_in,  # camera 坐标系下，X轴与Y轴过滤后的点云真值
                    'calib': calib,  # 相机内参 self.K[seq]
                    'tr_error': T,  # decalib 误差
                    'rot_error': R,
                    # 'seq': int(seq),  #以上一样
                    'img_path': img_path,
                    'rgb_name': rgb_name + '.png',
                    'item': item,
                    'extrin': RT,  # 真值
                    'initial_RT': initial_RT
                }  # self.val_RT[idx] 误差
        else:
            if not self.use_reflectance:
                sample = {
                    'rgb': img,
                    'point_cloud': pc_in,
                    'calib': calib,
                    'tr_error': T,
                    'rot_error': R,
                    # 'seq': int(seq),
                    'rgb_name': rgb_name,
                    'item': item,
                    'extrin': RT
                }
            else:
                sample = {
                    'reflectance': reflectance,
                    'rgb': img,
                    'point_cloud': pc_in,
                    'calib': calib,
                    'tr_error': T,
                    'rot_error': R,
                    # 'seq': int(seq),
                    'rgb_name': rgb_name,
                    'item': item,
                    'extrin': RT
                }

        return sample


class DatasetLidarCameraKittiOdometry(Dataset):
    """
    self.all_files = [] 获取所有文件列表 
    self.K = {} K内参 
    self.GTs_T_cam02_velo = {} 外参真值
    self.val_RT = [] T-init 读取/生成
    self.use_reflectance  使用反射率（强度信息）

    self.transform 未使用
    self.augmentation 未使用
    """

    def __init__(self,
                 dataset_dir,
                 transform=None,
                 augmentation=False,
                 use_reflectance=False,
                 max_t=1.5,
                 max_r=20.,
                 split='val',
                 device='cpu',
                 test_sequence='00',
                 suf='.png'):
        super(DatasetLidarCameraKittiOdometry, self).__init__()

        self.use_reflectance = use_reflectance
        self.maps_folder = ''
        self.device = device
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dir = dataset_dir
        self.transform = transform
        self.split = split
        self.test_sequence = test_sequence
        self.GTs_R = {}
        self.GTs_T = {}
        self.GTs_T_cam02_velo = {}
        self.K = {}
        self.suf = suf
        self.all_files = []
        self.val_RT = []
        print("zghself",self.max_r,self.max_t)
        self.sequence_list = [
            '00','01', '02', '03', '04', '05'
            , '06', '07', '08', '09', '10',
            #  '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'
        ]  # '00',# 00 is test

        #获取 内参、外参真值和all_files
        for seq in self.sequence_list:
            odom = odometry(self.root_dir, seq)
            calib = odom.calib
            T_cam02_velo_np = calib.T_cam2_velo  #gt pose from cam02 to velo_lidar (T_cam02_velo: 4x4)
            self.K[seq] = calib.K_cam2  # 3x3
            self.GTs_T_cam02_velo[
                seq] = T_cam02_velo_np  #gt pose from cam02 to velo_lidar (T_cam02_velo: 4x4)

            image_list = os.listdir(
                os.path.join(dataset_dir, 'sequences', seq, 'image_2'))
            image_list.sort()

            # 获取指定目录下的图像文件列表并按名称排序
            for image_name in image_list:
                if not os.path.exists(
                        os.path.join(dataset_dir, 'sequences', seq, 'velodyne',
                                     str(image_name.split('.')[0]) + '.bin')):
                    continue
                if not os.path.exists(
                        os.path.join(dataset_dir, 'sequences', seq, 'image_2',
                                     str(image_name.split('.')[0]) + suf)):
                    continue

                if seq == self.test_sequence and split == 'test':
                     self.all_files.append(os.path.join(seq,image_name.split('.')[0]))
            print("zghself.all_files",len(self.all_files))

        # T-init 读取/生成
        # self.val_RT = []
        if split == 'val' or split == 'test':
            val_RT_file = os.path.join(
                dataset_dir, 'sequences',
                f'val_RT_left_seq{test_sequence}_{max_r:.2f}_{max_t:.2f}.csv')
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
                    rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    transl_x = np.random.uniform(-max_t, max_t)
                    transl_y = np.random.uniform(-max_t, max_t)
                    transl_z = np.random.uniform(-max_t, max_t)
                    # transl_z = np.random.uniform(-max_t, min(max_t, 1.))
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
            #io.imshow(np.array(rgb))
            #io.show()

        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        item = self.all_files[idx]
        seq = str(item.split('/')[0])
        rgb_name = str(item.split('/')[1])
        img_path = os.path.join(self.root_dir, 'sequences', seq, 'image_2',
                                rgb_name + self.suf)
        lidar_path = os.path.join(self.root_dir, 'sequences', seq, 'velodyne',
                                  rgb_name + '.bin')
        lidar_scan = np.fromfile(lidar_path, dtype=np.float32)
        # 每行有4个元素。这样做是因为LIDAR扫描通常会生成(x, y, z, intensity)四个值作为一个点的信息。
        pc = lidar_scan.reshape((-1, 4))

        # 通过筛选条件创建一个布尔数组_indices，用于标记在x和y轴上超出范围[-3, 3]的点云。
        valid_indices = pc[:, 0] < -3.
        valid_indices = valid_indices | (pc[:, 0] > 3.)
        valid_indices = valid_indices | (pc[:, 1] < -3.)
        valid_indices = valid_indices | (pc[:, 1] > 3.)
        pc = pc[valid_indices].copy()
        # 筛选后的点云转换为torch.Tensor类型
        pc_org = torch.from_numpy(pc.astype(np.float32))
        # 是否使用强度信息
        if self.use_reflectance:
            reflectance = pc[:, 3].copy(
            )  # reflectance = pc[:, 3].copy()：这一行代码从变量pc中选择第4列的数据
            reflectance = torch.from_numpy(reflectance).float()
        RT = self.GTs_T_cam02_velo[seq].astype(np.float32)

        # 如果是，说明点云数据已经是所需的形状，不需要进行转换。
        if pc_org.shape[1] == 4 or pc_org.shape[1] == 3:
            pc_org = pc_org.t()
        # 代码检查点云数据的行数是否为3。如果是，说明点云数据是三维坐标（x、y、z），
        # 需要添加一个齐次坐标（homogeneous coordinate）来表示点的位置。
        if pc_org.shape[0] == 3:
            homogeneous = torch.ones(pc_org.shape[1]).unsqueeze(0)
            pc_org = torch.cat((pc_org, homogeneous), 0)
        elif pc_org.shape[0] == 4:
            # 这行代码检查点云数据的第四行是否全为1。如果不是，说明齐次坐标的值不正确，需要将其设置为1。
            if not torch.all(pc_org[3, :] == 1.):
                pc_org[3, :] = 1.
        else:
            raise TypeError("Wrong PointCloud shape")


## -----------------------------------------------------------------
        pc_rot = np.matmul(RT, pc_org.numpy())
        # cam坐标系下 lidar 真值
        pc_rot = pc_rot.astype(np.float32).copy()
        pc_gt = torch.from_numpy(pc_rot)
        h_mirror = False
        img = Image.open(img_path)
        img_rotation = 0.

        try:
            img = self.custom_transform(img, img_rotation, h_mirror)
        except OSError:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)


        if self.split == 'train':
            R = mathutils.Euler((radians(img_rotation), 0, 0), 'XYZ')
            T = mathutils.Vector((0., 0., 0.))
            pc_gt = rotate_forward(pc_gt, R, T)

        if self.split == 'train':
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle,
                                     max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle,
                                     max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle,
                                     max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, self.max_t)
            # transl_z = np.random.uniform(-self.max_t, min(self.max_t, 1.))
        else:
            initial_RT = self.val_RT[idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]

        # 随机设置一定范围内的标定参数扰动值
        # train的时候每次都随机生成,每个epoch使用不同的参数
        # test则在初始化的时候提前设置好,每个epoch都使用相同的参数
        R = mathutils.Euler((rotx, roty, rotz))
        T = mathutils.Vector((transl_x, transl_y, transl_z))

        R, T = invert_pose(R, T)
        R_tensor, T_tensor = torch.tensor(R), torch.tensor(T)

        calib = self.K[seq]
        if h_mirror:
            calib[2] = (img.shape[2] / 2) * 2 - calib[2]

        if self.split == 'test':
            if not self.use_reflectance:
                sample = {
                    'rgb': img,  # 归一化/torch 图像， 本实例未镜像
                    'point_cloud': pc_gt,  # camera 坐标系下，X轴与Y轴过滤后的点云真值
                    'pc_rotated': pc_rotated,  # camera 坐标系下，旋转过的点云
                    'calib': calib,  # 相机内参 self.K[seq]
                    'tr_error': T_tensor,  # decalib 误差
                    'rot_error': R_tensor,
                    'seq': int(seq),  #以上一样
                    'img_path': img_path,
                    'rgb_name': rgb_name + '.png',
                    'item': item,
                    'extrin': RT_zgh,  # 真值
                    'initial_RT': initial_RT
                }  # self.val_RT[idx] 误差
            else:
                sample = {
                    'reflectance': reflectance,
                    'rgb': img,  # 归一化/torch 图像， 本实例未镜像
                    'point_cloud': pc_gt,  # camera 坐标系下，X轴与Y轴过滤后的点云真值
                    # 'pc_rotated': pc_rotated,  # camera 坐标系下，旋转过的点云
                    'calib': calib,  # 相机内参 self.K[seq]
                    'tr_error': T_tensor,  # decalib 误差
                    'rot_error': R_tensor,
                    'seq': int(seq),  #以上一样
                    'img_path': img_path,
                    'rgb_name': rgb_name + '.png',
                    'item': item,
                    # 'extrin': RT_zgh,  # 真值
                    'initial_RT': initial_RT
                }  # self.val_RT[idx] 误差
        else:
            if not self.use_reflectance:
                sample = {
                    'rgb': img,
                    'point_cloud': pc_gt,  # camera 坐标系下，X轴与Y轴过滤后的点云真值
                    # 'pc_rotated': pc_rotated,  # camera 坐标系下，旋转过的点云
                    'calib': calib,
                    'tr_error': T,
                    'rot_error': R,
                    'seq': int(seq),
                    'rgb_name': rgb_name,
                    'item': item,
                    'extrin': RT
                }
            else:
                sample = {
                    'reflectance': reflectance,
                    'rgb': img,
                    'point_cloud': pc_gt,  # camera 坐标系下，X轴与Y轴过滤后的点云真值
                    # 'pc_rotated': pc_rotated,  # camera 坐标系下，旋转过的点云
                    'calib': calib,
                    'tr_error': T,
                    'rot_error': R,
                    'seq': int(seq),
                    'rgb_name': rgb_name,
                    'item': item,
                    'extrin': RT
                }

        return sample

EPOCH = 1


def _init_fn(worker_id, seed):
    """
    非常抱歉，我之前的回答可能引起了混淆。你是对的，在多进程数据加载情况下，
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


# def init_fn(x):
#     return _init_fn(x, seed)


def init_fn(x):
    seed = torch.initial_seed() % 2**32
    return _init_fn(x, seed)


def fetch_dataloader(args, split='train'):
    if args.stage == 'kitti':
        print("This kitti-odometry dataset!")
        dataset_class = DatasetLidarCameraKittiOdometry
        dataset_kitti = dataset_class(args.data_folder,
                                      max_r=args.max_r,
                                      max_t=args.max_t,
                                      split=split,
                                      use_reflectance=args.use_reflectance,
                                      test_sequence=args.test_sequence)
        dataset_kitti_size = len(dataset_kitti)
        print('Number of the {} Dataset: {}'.format(split, dataset_kitti_size))

    if split == 'train':
        shuffle = True
    else:
        shuffle = False

    dataloader_kitti = data.DataLoader(dataset=dataset_kitti,
                                       shuffle=shuffle,
                                       batch_size=args.batch_size,
                                       num_workers=args.num_worker,
                                       worker_init_fn=init_fn,
                                       collate_fn=merge_inputs,
                                       drop_last=False,
                                       pin_memory=True)
    dataloader_kitti_size = len(dataloader_kitti)
    print('Number of the {} DataLoader: {}'.format(split,
                                                   dataloader_kitti_size))

    return dataloader_kitti, dataset_kitti_size

