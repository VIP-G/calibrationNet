import torch
import torch.nn.functional as F
import numpy as np
import numpy
from scipy import interpolate

import mathutils
import math

# dataloader
from torch.utils.data.dataloader import default_collate
from matplotlib import cm

# 测试用的 显示tensor格式图像
import matplotlib.pyplot as plt


# DatasetLidarCamera -- 获取 decalib误差值
def get_decalib(max_r, max_t):
    rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
    roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
    rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
    transl_x = np.random.uniform(-max_t, max_t)
    transl_y = np.random.uniform(-max_t, max_t)
    transl_z = np.random.uniform(-max_t, max_t)
    return transl_x, transl_y, transl_z, rotx, roty, rotz


# DatasetLidarCamera --- 生成深度图 get_2D_lidar_projection，lidar_project_depth
def get_2D_lidar_projection(pcl, cam_intrinsic):
    # 在Python中，@ 符号被用作矩阵乘法运算符。
    pcl_xyz = cam_intrinsic @ pcl.T
    pcl_xyz = pcl_xyz.T
    pcl_z = pcl_xyz[:, 2]
    pcl_xyz = pcl_xyz / (pcl_xyz[:, 2, None] + 1e-10)
    pcl_uv = pcl_xyz[:, :2]

    return pcl_uv, pcl_z


def lidar_project_depth(pc_rotated, cam_calib, img_shape, reflectance=None):
    # 将pc_rotated转换为numpy数组
    pc_rotated = pc_rotated[:3, :].detach().cpu().numpy()
    # 获取相机内参
    cam_intrinsic = cam_calib  #.numpy()
    # 获取点云在图像上的投影
    pcl_uv, pcl_z = get_2D_lidar_projection(pc_rotated.T, cam_intrinsic)  
    # 创建一个掩码，用于筛选出在图像范围内的点云
    mask = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) & (
        pcl_uv[:, 1] > 0) & (pcl_uv[:, 1] < img_shape[0]) & (pcl_z > 0)
    # 筛选出在图像范围内的点云
    pcl_uv = pcl_uv[mask]
    pcl_z = pcl_z[mask]
    # 将pcl_uv转换为uint32类型
    pcl_uv = pcl_uv.astype(np.uint32)
    # 将pcl_z转换为1列
    pcl_z = pcl_z.reshape(-1, 1)

    # 创建一个全零的深度图像
    depth_img = np.zeros((img_shape[0], img_shape[1], 1))
    # 将点云的深度值赋给深度图像
    depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = pcl_z
    # 将深度图像转换为torch张量
    depth_img = torch.from_numpy(depth_img.astype(np.float32))
    # 将深度图像的维度从HWC转换为CHW
    depth_img = depth_img.permute(2, 0, 1)
    # 初始化反射图像
    reflect_img = None
    # 如果有反射值，则进行处理
    if reflectance is not None:
        # 筛选出在图像范围内的反射值
        reflectance = reflectance[mask]
        # 将反射值转换为1列
        reflectance = reflectance.reshape(-1, 1)
        # 创建一个全零的反射图像
        reflect_img = np.zeros((img_shape[0], img_shape[1], 1))
        # 将反射值赋给反射图像
        reflect_img[pcl_uv[:, 1], pcl_uv[:, 0]] = reflectance
        # 将反射图像转换为torch张量
        reflect_img = torch.from_numpy(reflect_img.astype(np.float32))
        # 将反射图像的维度从HWC转换为CHW
        reflect_img = reflect_img.permute(2, 0, 1)

    # 返回深度图像、点云在图像上的投影和反射图像
    return depth_img, pcl_uv, reflect_img

# DatasetLidarCamera --- 旋转点云 rotate_points,rotate_points_torch,rotate_forward,rotate_back
def rotate_points(PC, R, T=None, inverse=True):
    if T is not None:
        R = R.to_matrix()
        R.resize_4x4()
        T = mathutils.Matrix.Translation(T)
        RT = T * R
    else:
        RT = R.copy()
    if inverse:
        RT.invert_safe()
    # RT = torch.tensor(RT, device='cpu', dtype=torch.float)
    # PC = PC.to('cpu')
    # print(type(RT))
    RT = torch.tensor(RT, device=PC.device, dtype=torch.float)  # Hliu

    if PC.shape[0] == 4:
        PC = torch.mm(RT, PC)
    elif PC.shape[1] == 4:
        PC = torch.mm(RT, PC.t())
        PC = PC.t()
    else:
        raise TypeError(
            "Point cloud must have shape [Nx4] or [4xN] (homogeneous coordinates)"
        )
    return PC


def rotate_points_torch(PC, R, T=None, inverse=True):
    if T is not None:
        R = quat2mat(R)
        T = tvector2mat(T)
        RT = torch.mm(T, R)
    else:
        RT = R.clone()
    if inverse:
        RT = RT.inverse()
    RT = RT.to(PC.device)
    if PC.shape[0] == 4:
        PC = torch.mm(RT, PC)
    elif PC.shape[1] == 4:
        PC = torch.mm(RT, PC.t())
        PC = PC.t()
    else:
        raise TypeError(
            "Point cloud must have shape [Nx4] or [4xN] (homogeneous coordinates)"
        )
    return PC


def rotate_forward(PC, R, T=None):
    """
    Transform the point cloud PC, so to have the points 'as seen from' the new
    pose T*R
    Args:
        PC (torch.Tensor): Point Cloud to be transformed, shape [4xN] or [Nx4]
        R (torch.Tensor/mathutils.Euler): can be either:
            * (mathutils.Euler) euler angles of the rotation part, in this case T cannot be None
            * (torch.Tensor shape [4]) quaternion representation of the rotation part, in this case T cannot be None
            * (mathutils.Matrix shape [4x4]) Rotation matrix,
                in this case it should contains the translation part, and T should be None
            * (torch.Tensor shape [4x4]) Rotation matrix,
                in this case it should contains the translation part, and T should be None
        T (torch.Tensor/mathutils.Vector): Translation of the new pose, shape [3], or None (depending on R)

    Returns:
        torch.Tensor: Transformed Point Cloud 'as seen from' pose T*R
    """
    if isinstance(R, torch.Tensor):
        return rotate_points_torch(PC, R, T, inverse=True)
    else:
        return rotate_points(PC, R, T, inverse=True)


def rotate_back(PC_ROTATED, R, T=None):
    """
    Inverse of :func:`~utils.rotate_forward`.
    """
    if isinstance(R, torch.Tensor):
        return rotate_points_torch(PC_ROTATED, R, T, inverse=False)
    else:
        return rotate_points(PC_ROTATED, R, T, inverse=False)


def show_tensor_img(image_tensor, name='1'):
    # image_tensor = image_tensor * 255
    image_np = image_tensor.cpu().numpy()
    plt.imshow(image_np[0], cmap='gray')
    # 可选：添加标题等图像信息
    plt.title(name)
    plt.axis("off")  # 不显示坐标轴

    # 显示图像
    plt.show()


def show_tensor_imgs(image_tensor,
                     image_tensor1,
                     image_tensor2,
                     image_tensor3,
                     name='1'):
    # image_tensor = image_tensor * 255
    image_np = image_tensor.cpu().numpy()
    image_np1 = image_tensor1.cpu().numpy()
    image_np2 = image_tensor2.cpu().numpy()
    image_np3 = image_tensor3.cpu().numpy()

    plt.subplot(2, 2, 1)  # 第一个子图
    plt.imshow(image_np[0], cmap='gray')
    plt.subplot(2, 2, 2)  # 第一个子图
    plt.imshow(image_np1[0], cmap='gray')
    plt.subplot(2, 2, 3)  # 第一个子图
    plt.imshow(image_np2[0], cmap='gray')
    plt.subplot(2, 2, 4)  # 第一个子图
    plt.imshow(image_np3[0], cmap='gray')

    plt.title(name)
    plt.axis("off")  # 不显示坐标轴

    # 显示图像
    plt.show()


# Dataloader
def merge_inputs(queries):
    imgs = []
    lidar_input = []
    point_clouds = []
    depth_gt = []
    returns = {
        key: default_collate([d[key] for d in queries])
        for key in queries[0] if key != 'point_cloud' and key != 'rgb'
        and key != 'lidar_input' and key != 'depth_gt'
    }
    for input in queries:
        point_clouds.append(input['point_cloud'])
        imgs.append(input['rgb'])
        lidar_input.append(input['lidar_input'])
        depth_gt.append(input['depth_gt'])
    returns['point_cloud'] = point_clouds
    returns['rgb'] = torch.stack(imgs)
    returns['lidar_input'] = torch.stack(lidar_input)
    returns['depth_gt'] = torch.stack(depth_gt)
    return returns


# 原来 LCCNet 网路 merge_inputs
def merge_inputs_copy(queries):
    point_clouds = []
    imgs = []
    reflectances = []
    returns = {
        key: default_collate([d[key] for d in queries])
        for key in queries[0]
        if key != 'point_cloud' and key != 'rgb' and key != 'reflectance'
    }
    for input in queries:
        point_clouds.append(input['point_cloud'])
        imgs.append(input['rgb'])
        if 'reflectance' in input:
            reflectances.append(input['reflectance'])
    returns['point_cloud'] = point_clouds
    returns['rgb'] = imgs
    if len(reflectances) > 0:
        returns['reflectance'] = reflectances
    return returns


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [
                pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2,
                pad_ht - pad_ht // 2
            ]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy

    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata((x1, y1),
                                  dx, (x0, y0),
                                  method='nearest',
                                  fill_value=0)

    flow_y = interpolate.griddata((x1, y1),
                                  dy, (x0, y0),
                                  method='nearest',
                                  fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    
    xgrid, ygrid = coords.split([1, 1], dim=-1)

    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)
   
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


# 通过重复扩展坐标网格，生成一个形状为(batch, 2, ht, wd)的张量，
def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device),
                            torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(
        flow, size=new_size, mode=mode, align_corners=True)


# ----------------------------------- LCCNet
import numpy as np  
from scipy.spatial.transform import Rotation as R  
  
def decompose_matrix(matrix):  
    # 假设matrix是一个4x4的列表（或元组）的列表（或元组）  
    assert len(matrix) == 4 and all(len(row) == 4 for row in matrix), "Matrix must be 4x4"  
  
    # 提取平移分量（已经是0，但按常规提取）  
    translation = np.array(matrix[:3, 3], dtype=float)  # 注意：这里实际上应该是matrix[3][:3]，但您的数据是[0, 0, 0]  
  
    # 提取3x3的旋转+缩放部分  
    rotation_scale_matrix = np.array(matrix[:3, :3], dtype=float)  
  
    # 计算缩放分量（这里我们简单地取每列向量的长度，但注意非均匀缩放和旋转的复杂性）  
    scale = np.linalg.norm(rotation_scale_matrix, axis=0)  
  
    # 注意：由于缩放和旋转的耦合，这里的scale可能不是纯粹的缩放因子  
    # 但对于接近单位矩阵的矩阵，这通常是一个合理的近似  
  
    # 归一化以去除缩放的影响，但保留旋转  
    # 注意：这里我们直接除以scale，这可能在非均匀缩放时导致问题  
    # 一个更健壮的方法是使用SVD分解，但这里我们保持简单  
    normalized_rotation_matrix = rotation_scale_matrix / scale  
  
    # 由于NumPy没有直接的四元数表示，我们通常不会在这里计算四元数  
    # 但如果需要，可以使用其他库（如scipy.spatial.transform.Rotation）  
    # 或者手动实现四元数的计算（这通常涉及更多的数学）  
  
    # 由于我们没有计算四元数，我们可以简单地返回归一化后的旋转矩阵作为“旋转”部分  
    # 但请注意，这通常不是最佳做法，因为它没有提供旋转的直观表示  
    rotation_matrix = normalized_rotation_matrix  
  
    # 返回平移、旋转矩阵（不是四元数）和缩放  
    return translation, rotation_matrix, scale

def invert_pose(R, T):
    """
    Given the 'sampled pose' (aka H_init), we want CMRNet to predict inv(H_init).
    inv(T*R) will be used as ground truth for the network.
    Args:
        R (mathutils.Euler): Rotation of 'sampled pose'
        T (mathutils.Vector): Translation of 'sampled pose'

    Returns:
        (R_GT, T_GT) = (mathutils.Quaternion, mathutils.Vector)
    """
    R = R.to_matrix()
    R.resize_4x4()
    T = mathutils.Matrix.Translation(T)
    RT = T @ R
    RT.invert_safe()
    T_GT, R_GT, _ = RT.decompose()
    return R_GT.normalized(), T_GT



def quaternion_from_matrix(matrix):
    """
    Convert a rotation matrix to quaternion.
    Args:
        matrix (torch.Tensor): [4x4] transformation matrix or [3,3] rotation matrix.

    Returns:
        torch.Tensor: shape [4], normalized quaternion
    """
    if matrix.shape == (4, 4):
        R = matrix[:-1, :-1]
    elif matrix.shape == (3, 3):
        R = matrix
    else:
        raise TypeError("Not a valid rotation matrix")
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    q = torch.zeros(4, device=matrix.device)
    if tr > 0.:
        S = (tr + 1.0).sqrt() * 2
        q[0] = 0.25 * S
        q[1] = (R[2, 1] - R[1, 2]) / S
        q[2] = (R[0, 2] - R[2, 0]) / S
        q[3] = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = (1.0 + R[0, 0] - R[1, 1] - R[2, 2]).sqrt() * 2
        q[0] = (R[2, 1] - R[1, 2]) / S
        q[1] = 0.25 * S
        q[2] = (R[0, 1] + R[1, 0]) / S
        q[3] = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = (1.0 + R[1, 1] - R[0, 0] - R[2, 2]).sqrt() * 2
        q[0] = (R[0, 2] - R[2, 0]) / S
        q[1] = (R[0, 1] + R[1, 0]) / S
        q[2] = 0.25 * S
        q[3] = (R[1, 2] + R[2, 1]) / S
    else:
        S = (1.0 + R[2, 2] - R[0, 0] - R[1, 1]).sqrt() * 2
        q[0] = (R[1, 0] - R[0, 1]) / S
        q[1] = (R[0, 2] + R[2, 0]) / S
        q[2] = (R[1, 2] + R[2, 1]) / S
        q[3] = 0.25 * S
    return q / q.norm()


def quatmultiply(q, r):
    """
    Multiply two quaternions
    Args:
        q (torch.Tensor/nd.ndarray): shape=[4], first quaternion
        r (torch.Tensor/nd.ndarray): shape=[4], second quaternion

    Returns:
        torch.Tensor: shape=[4], normalized quaternion q*r
    """
    t = torch.zeros(4, device=q.device)
    t[0] = r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3]
    t[1] = r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2]
    t[2] = r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1]
    t[3] = r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0]
    return t / t.norm()


def quat2mat(q):
    """
    Convert a quaternion to a rotation matrix
    Args:
        q (torch.Tensor): shape [4], input quaternion

    Returns:
        torch.Tensor: [4x4] homogeneous rotation matrix
    """
    assert q.shape == torch.Size([4]), "Not a valid quaternion"
    if q.norm() != 1.:
        q = q / q.norm()
    mat = torch.zeros((4, 4), device=q.device)
    mat[0, 0] = 1 - 2 * q[2]**2 - 2 * q[3]**2
    mat[0, 1] = 2 * q[1] * q[2] - 2 * q[3] * q[0]
    mat[0, 2] = 2 * q[1] * q[3] + 2 * q[2] * q[0]
    mat[1, 0] = 2 * q[1] * q[2] + 2 * q[3] * q[0]
    mat[1, 1] = 1 - 2 * q[1]**2 - 2 * q[3]**2
    mat[1, 2] = 2 * q[2] * q[3] - 2 * q[1] * q[0]
    mat[2, 0] = 2 * q[1] * q[3] - 2 * q[2] * q[0]
    mat[2, 1] = 2 * q[2] * q[3] + 2 * q[1] * q[0]
    mat[2, 2] = 1 - 2 * q[1]**2 - 2 * q[2]**2
    mat[3, 3] = 1.
    return mat


def tvector2mat(t):
    """
    Translation vector to homogeneous transformation matrix with identity rotation
    Args:
        t (torch.Tensor): shape=[3], translation vector

    Returns:
        torch.Tensor: [4x4] homogeneous transformation matrix

    """
    assert t.shape == torch.Size([3]), "Not a valid translation"
    mat = torch.eye(4, device=t.device)
    mat[0, 3] = t[0]
    mat[1, 3] = t[1]
    mat[2, 3] = t[2]
    return mat


def mat2xyzrpy(rotmatrix):
    """
    Decompose transformation matrix into components
    Args:
        rotmatrix (torch.Tensor/np.ndarray): [4x4] transformation matrix

    Returns:
        torch.Tensor: shape=[6], contains xyzrpy
    """
    roll = math.atan2(-rotmatrix[1, 2], rotmatrix[2, 2])
    pitch = math.asin(rotmatrix[0, 2])
    yaw = math.atan2(-rotmatrix[0, 1], rotmatrix[0, 0])
    x = rotmatrix[:3, 3][0]
    y = rotmatrix[:3, 3][1]
    z = rotmatrix[:3, 3][2]

    return torch.tensor([x, y, z, roll, pitch, yaw],
                        device=rotmatrix.device,
                        dtype=rotmatrix.dtype)


def to_rotation_matrix(R, T):
    R = quat2mat(R)
    T = tvector2mat(T)
    RT = torch.mm(T, R)
    return RT


def overlay_imgs(rgb, lidar, idx=0):
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]

    rgb = rgb.clone().cpu().permute(1, 2, 0).numpy()
    rgb = rgb * std + mean
    lidar = lidar.clone()

    lidar[lidar == 0] = 1000.
    lidar = -lidar
    #lidar = F.max_pool2d(lidar, 3, 1, 1)
    lidar = F.max_pool2d(lidar, 3, 1, 1)
    lidar = -lidar
    lidar[lidar == 1000.] = 0.

    #lidar = lidar.squeeze()
    lidar = lidar[0][0]
    lidar = (lidar * 255).int().cpu().numpy()
    lidar_color = cm.jet(lidar)
    lidar_color[:, :, 3] = 0.5
    lidar_color[lidar == 0] = [0, 0, 0, 0]
    blended_img = lidar_color[:, :, :3] * (np.expand_dims(lidar_color[:, :, 3], 2)) + \
                  rgb * (1. - np.expand_dims(lidar_color[:, :, 3], 2))
    blended_img = blended_img.clip(min=0., max=1.)
    #io.imshow(blended_img)
    #io.show()
    #plt.figure()
    #plt.imshow(blended_img)
    #io.imsave(f'./IMGS/{idx:06d}.png', blended_img)
    return blended_img

def overlay_reflect_imgs(rgb, lidar, idx=0):
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]

    rgb = rgb.clone().cpu().permute(1, 2, 0).numpy()
    rgb = rgb * std + mean

    lidar = lidar[0,0:1, :, :].clone()
    # lidar = lidar[0,1:2, :, :].clone()
    lidar = lidar.unsqueeze(0)

    lidar[lidar == 0] = 1000.
    lidar = -lidar
    #lidar = F.max_pool2d(lidar, 3, 1, 1)
    lidar = F.max_pool2d(lidar, 3, 1, 1)
    lidar = -lidar
    lidar[lidar == 1000.] = 0.

    #lidar = lidar.squeeze()
    lidar = lidar[0][0]
    lidar = (lidar * 255).int().cpu().numpy()
    lidar_color = cm.jet(lidar)
    lidar_color[:, :, 3] = 0.5
    lidar_color[lidar == 0] = [0, 0, 0, 0]
    # cv2.imshow("11",lidar_color[:, :, [2, 1, 0]])
    blended_img = lidar_color[:, :, :3] * (np.expand_dims(lidar_color[:, :, 3], 2)) + \
                  rgb * (1. - np.expand_dims(lidar_color[:, :, 3], 2))
    blended_img = blended_img.clip(min=0., max=1.)
    #io.imshow(blended_img)
    #io.show()
    #plt.figure()
    # plt.imshow(lidar_color)
    # plt.imshow(blended_img)
    #io.imsave(f'./IMGS/{idx:06d}.png', blended_img)
    return blended_img