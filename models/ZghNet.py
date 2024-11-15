import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.update import BasicUpdateBlock,BasicUpdateBlock0926
from models.extractor import *
from models.modefy_model import *
from models.corr import CorrBlock
from models.utils import  coords_grid

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:

        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass

# 原始版本
class ZghNet(nn.Module):
    
    def __init__(self, args):
        super(ZghNet, self).__init__()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4
        if 'dropout' not in self.args:
            self.args.dropout = 0
        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        self.f_rgb = BasicEncoder(output_dim=256,
                                  norm_fn='instance',
                                  sensor='cam',
                                  dropout=args.dropout)
        self.f_lidar = BasicEncoder(output_dim=256,
                                    norm_fn='instance',
                                    sensor='lidar',
                                    use_reflectance=args.use_reflectance,
                                    dropout=args.dropout)
        self.cnet = BasicEncoder(output_dim=hdim + cdim,
                                 norm_fn='batch',
                                 dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

        fc_size = 4096  #4096  #缩略图 #原图#15360  #暂时 debug 出来的

        self.leakyRELU = nn.LeakyReLU(0.1)
        self.fc1 = nn.Linear(fc_size, 512)
        self.fc1_trasl = nn.Linear(512, 256)
        self.fc1_rot = nn.Linear(512, 256)
        self.fc2_trasl = nn.Linear(256, 3)
        self.fc2_rot = nn.Linear(256, 4)
        self.dropout = nn.Dropout(args.dropout)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self,
                rgb_img,
                depth_img,
                iters=12, # 12
                flow_init=None,
                upsample=True,
                test_mode=False):
        """ Estimate optical flow between pair of frames """

        rgb_img = 2 * (rgb_img / 255.0) - 1.0
        # depth_img = 2 * (depth_img / 255.0) - 1.0
        depth_img = 2 * depth_img - 1.0


        rgb_img = rgb_img.contiguous()
        depth_img = depth_img.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            # fmap1, fmap2 = self.fnet([image1, image2])
            fmap1 = self.f_rgb(rgb_img)
            fmap2 = self.f_lidar(depth_img)

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(rgb_img)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(rgb_img)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

        #Hliu 改的全连接层
        x = coords1.view(coords1.shape[0], -1)
        x = self.dropout(x)
        x = self.leakyRELU(self.fc1(x))

        transl = self.leakyRELU(self.fc1_trasl(x))
        rot = self.leakyRELU(self.fc1_rot(x))
        transl = self.fc2_trasl(transl)
        rot = self.fc2_rot(rot)
        rot = F.normalize(rot, dim=1)

        return transl, rot

# 加入多头注意力
class EnMutiAttenNet(nn.Module):
    
    def __init__(self, args):
        super(EnMutiAttenNet, self).__init__()
        self.args = args

        self.hidden_dim = hdim = 192
        self.context_dim = cdim = 192
        args.corr_levels = 4
        args.corr_radius = 4
        if 'dropout' not in self.args:
            self.args.dropout = 0
        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        self.f_rgb = EncoderMultiAttention(output_dim=384,
                                  norm_fn='instance',
                                  sensor='cam',
                                  dropout=args.dropout)
        self.f_lidar = EncoderMultiAttention(output_dim=384,
                                    norm_fn='instance',
                                    sensor='lidar',
                                    use_reflectance=args.use_reflectance,
                                    dropout=args.dropout)
        self.cnet = EncoderMultiAttention(output_dim=hdim + cdim,
                                 norm_fn='batch',
                                 dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

        fc_size = 1024  #4096  #缩略图 #原图#15360  #暂时 debug 出来的
        fc_size_1 = 512
        fc_size_2 = 128
        self.leakyRELU = nn.LeakyReLU(0.1)
        self.fc1 = nn.Linear(fc_size, fc_size_1)
        self.fc1_trasl = nn.Linear(fc_size_1, fc_size_2)
        self.fc1_rot = nn.Linear(fc_size_1, fc_size_2)
        self.fc2_trasl = nn.Linear(fc_size_2, 3)
        self.fc2_rot = nn.Linear(fc_size_2, 4)
        self.dropout = nn.Dropout(args.dropout)

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        key = 16
        coords0 = coords_grid(N, H // key, W // key, device=img.device)
        coords1 = coords_grid(N, H // key, W // key, device=img.device)

        return coords0, coords1

    def forward(self,
                rgb_img,
                depth_img,
                iters=12, ):
        """ Estimate optical flow between pair of frames """

        rgb_img = 2 * (rgb_img / 255.0) - 1.0
        # depth_img = 2 * (depth_img / 255.0) - 1.0
        depth_img = 2 * depth_img - 1.0

        rgb_img = rgb_img.contiguous()
        depth_img = depth_img.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            # fmap1, fmap2 = self.fnet([image1, image2])
            fmap1 = self.f_rgb(rgb_img)
            fmap2 = self.f_lidar(depth_img)

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(rgb_img)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(rgb_img)

        flow_predictions = []
        for itr in range(16):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

        #Hliu 改的全连接层
        x = coords1.view(coords1.shape[0], -1)
        x = self.dropout(x)
        x = self.leakyRELU(self.fc1(x))

        transl = self.leakyRELU(self.fc1_trasl(x))
        rot = self.leakyRELU(self.fc1_rot(x))
        transl = self.fc2_trasl(transl)
        rot = self.fc2_rot(rot)
        rot = F.normalize(rot, dim=1)

        return transl, rot

#使用可变卷积vision transformer
class vision_transformer(nn.Module):
    
    def __init__(self, args,
                 patch_size=4, num_classes=1000, expansion=4,
                 dim_stem=64, dims=[64, 128, 256, 512], depths=[2, 2, 4, 2], 
                 heads=[4, 8, 4, 8], 
                 window_sizes=[4, 4, 4, 4],
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, 
                 strides=[1,1,1,1], offset_range_factor=[1, 2, 3, 4], 
                 stage_spec=[['L', 'D'], ['L', 'D'], ['L', 'D', 'L', 'D'], ['L', 'D']], 
                 groups=[1, 1, 4, 6],
                 use_pes=[False, False, False, False], 
                 dwc_pes=[False, False, False, False],
                 sr_ratios=[8, 4, 2, 1], 
                 fixed_pes=[False, False, False, False],
                 no_offs=[False, False, False, False],
                 ns_per_pts=[4, 4, 4, 4],
                 use_dwc_mlps=[False, False, False, False],
                 ):
        super(vision_transformer, self).__init__()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4
        if 'dropout' not in self.args:
            self.args.dropout = 0
        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False
        self.img_shape = np.array(args.img_shape)// patch_size
        # img_size = img_size // patch_size
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.stages = nn.ModuleList()
        for i in range(3):
            dim1 = dim_stem if i == 0 else dims[i - 1] * 2
            dim2 = dims[i]
            self.stages.append(
                TransformerStage(self.img_shape, window_sizes[i], ns_per_pts[i],
                dim1, dim2, depths[i], stage_spec[i], groups[i], use_pes[i], 
                sr_ratios[i], heads[i], strides[i], 
                offset_range_factor[i], i,
                dwc_pes[i], no_offs[i], fixed_pes[i],
                attn_drop_rate, drop_rate, expansion, drop_rate, 
                dpr[sum(depths[:i]):sum(depths[:i + 1])],
                use_dwc_mlps[i])
            )
            self.img_shape = self.img_shape // 2
        
        self.down_projs = nn.ModuleList()
        for i in range(2):
            self.down_projs.append(
                nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1, bias=False),
                    LayerNormProxy(dims[i + 1])
                ) 
            )
           
        self.patch_proj1 = nn.Sequential(
            nn.Conv2d(3, dim_stem, 7, patch_size, 3),
            LayerNormProxy(dim_stem)
        )
        self.patch_proj2 = nn.Sequential(
            nn.Conv2d(2, dim_stem, 7, patch_size, 3),
            LayerNormProxy(dim_stem)
        )

        self.cnet = BasicEncoder(output_dim=256,
                                 norm_fn='batch',
                                 dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=128)

        fc_size = 1024  #4096  #缩略图 #原图#15360  #暂时 debug 出来的
        fc_size_1 = 512
        fc_size_2 = 128
        self.leakyRELU = nn.LeakyReLU(0.1)
        self.fc1 = nn.Linear(fc_size, fc_size_1)
        self.fc1_trasl = nn.Linear(fc_size_1, fc_size_2)
        self.fc1_rot = nn.Linear(fc_size_1, fc_size_2)
        self.fc2_trasl = nn.Linear(fc_size_2, 3)
        self.fc2_rot = nn.Linear(fc_size_2, 4)
        self.dropout = nn.Dropout(args.dropout)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 16, W // 16, device=img.device)
        coords1 = coords_grid(N, H // 16, W // 16, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def forward(self,
                rgb_img,
                depth_img,
                iters=12):
        """ Estimate optical flow between pair of frames """

        rgb_img = 2 * (rgb_img / 255.0) - 1.0
        # depth_img = 2 * (depth_img / 255.0) - 1.0
        depth_img = 2 * depth_img - 1.0


        rgb_img = rgb_img.contiguous()
        depth_img = depth_img.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        fmap1 = self.patch_proj1(rgb_img)
        fmap2 = self.patch_proj2(depth_img)

        for i in range(3):

            fmap1 = self.stages[i](fmap1)
            fmap2 = self.stages[i](fmap2)
            if i < 2:
                fmap1 = self.down_projs[i](fmap1)
                fmap2 = self.down_projs[i](fmap2)

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(rgb_img)
            net, inp = torch.split(cnet, [128, 128], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(rgb_img)


        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

        #Hliu 改的全连接层
        x = coords1.view(coords1.shape[0], -1)
        x = self.dropout(x)
        x = self.leakyRELU(self.fc1(x))

        transl = self.leakyRELU(self.fc1_trasl(x))
        rot = self.leakyRELU(self.fc1_rot(x))
        transl = self.fc2_trasl(transl)
        rot = self.fc2_rot(rot)
        rot = F.normalize(rot, dim=1)

        return transl, rot

# 0927使用GMA修改
# Gma和SKFlow都是RAFT的修改版，只是UpdateBlock不同
class UpdateGmaNet(nn.Module):
    
    def __init__(self, args):
        super(UpdateGmaNet, self).__init__()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        self.rgb = BasicEncoder(output_dim=256,sensor='cam', norm_fn='instance', dropout=args.dropout)
        self.lidar = BasicEncoder(output_dim=256,sensor='lidar', norm_fn='instance', dropout=args.dropout,use_reflectance=args.use_reflectance)
        self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = GMAUpdateBlock(self.args, hidden_dim=hdim)
        # self.update_block =SKUpdateBlock6_Deep_nopoolres_AllDecoder(self.args, hidden_dim=hdim)
        self.att = Attention(args=self.args, dim=cdim, heads=1, max_pos_size=160, dim_head=cdim)


        fc_size = 4096  #4096  #缩略图 #原图#15360  #暂时 debug 出来的

        self.leakyRELU = nn.LeakyReLU(0.1)
        self.fc1 = nn.Linear(fc_size, 512)
        self.fc1_trasl = nn.Linear(512, 256)
        self.fc1_rot = nn.Linear(512, 256)
        self.fc2_trasl = nn.Linear(256, 3)
        self.fc2_rot = nn.Linear(256, 4)
        self.dropout = nn.Dropout(args.dropout)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)

    
        return coords0, coords1


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1 = self.rgb(image1)
            fmap2 = self.lidar(image2)

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
            # attention, att_c, att_p = self.att(inp)
            attention = self.att(inp)

        coords0, coords1 = self.initialize_flow(image1)

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow, attention)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

        #Hliu 改的全连接层
        x = coords1.view(coords1.shape[0], -1)
        x = self.dropout(x)
        x = self.leakyRELU(self.fc1(x))

        transl = self.leakyRELU(self.fc1_trasl(x))
        rot = self.leakyRELU(self.fc1_rot(x))
        transl = self.fc2_trasl(transl)
        rot = self.fc2_rot(rot)
        rot = F.normalize(rot, dim=1)

        return transl, rot
