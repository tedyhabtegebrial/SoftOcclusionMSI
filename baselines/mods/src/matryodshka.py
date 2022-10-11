import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm
from .spt_utils import Utils as SptUtils

class ConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, k_size=3, dilation=1, stride=1, pad=1,
                 relu=True, v_coord=True, layer_norm=True, size=None):
        super(ConvBlock, self).__init__()
        v_chans = 1 if v_coord else 0
        self.conv = nn.Conv2d(v_chans + in_chans, out_chans,
                              k_size, dilation=dilation, stride=stride, padding=pad, bias=False)
        self.non_linearity = F.relu if relu else None
        self.conv.weight.data = torch.from_numpy(
            truncnorm.rvs(-1, 1, size=self.conv.weight.data.shape)).float()
        self.layer_norm = nn.LayerNorm(
            size, elementwise_affine=False) if layer_norm else None

    def forward(self, feats, v_coord=None):
        feats = feats if v_coord is None else torch.cat([feats, v_coord], dim=1)
        feats = self.conv(feats)
        if not self.layer_norm is None:
            feats = self.layer_norm(feats)
        if not (self.non_linearity is None):
            feats = self.non_linearity(feats)
        return feats

class FractionalConvolution(nn.Module):
    def __init__(self, in_chans, out_chans, k_size, dilation=1, stride=2, pad=1,
                    output_padding=1, relu=True, v_coord=True, layer_norm=True, size=None):
        super(FractionalConvolution, self).__init__()
        v_chans = 1 if v_coord else 0
        self.conv = nn.ConvTranspose2d(in_chans+v_chans, out_chans, stride=stride, kernel_size=k_size,
                                       output_padding=output_padding, padding=pad, dilation=dilation)
        self.non_linearity = F.relu if relu else None
        self.conv.weight.data = torch.from_numpy(
            truncnorm.rvs(-1, 1, size=self.conv.weight.data.shape)).float()
        self.layer_norm = nn.LayerNorm(
            (size[0], size[1]), elementwise_affine=False) if layer_norm else None

    def forward(self, feats, v_coord=None):
        feats = feats if v_coord is None else torch.cat([feats, v_coord], dim=1)
        feats = self.conv(feats)
        if not self.layer_norm is None:
            feats = self.layer_norm(feats)
        if not (self.non_linearity is None):
            feats = self.non_linearity(feats)
        return feats


class MatryODSHKhaNet(nn.Module):
    def __init__(self, configs):
        super(MatryODSHKhaNet, self).__init__()
        self.configs = configs
        self.spt_utls = SptUtils(configs)
        self.num_spheres = configs.num_spheres
        v_size = 1
        v_coord =  True
        self.inp_size = 2*self.num_spheres*3
        enc_i_chans = [self.inp_size, 64, 128, 128, 256, 256, 256, 512, 512, 512]
        self.enc_o_factors = [1, 2, 2, 4, 4, 4, 8, 8, 8, 8]
        self.enc_i_factors = [1, 1, 2, 2, 4, 4, 4, 8, 8, 8]
        v_coords = self.spt_utls.get_vcoords()
        v_coord_list = self._get_mul_scale_v_coords(v_coords.permute(0, 3, 1, 2))
        for idx,v in enumerate(v_coord_list):
            setattr(self, f'v_coord_{idx}', v)
        stride = [1, 2, 1, 2, 1, 1, 2, 1, 1, 1]
        enc_o_sizes = [[self.configs.height//i, self.configs.width//i] for i in self.enc_o_factors]
        self.conv_1_1 = ConvBlock(enc_i_chans[0], 64, size=(enc_o_sizes[0]), v_coord=v_coord, stride=stride[0])
        self.conv_1_2 = ConvBlock(64, 128, size=enc_o_sizes[1], v_coord=v_coord, stride=stride[1])
        self.conv_2_1 = ConvBlock(128, 128, size=enc_o_sizes[1], v_coord=v_coord, stride=stride[2])
        self.conv_2_2 = ConvBlock(128, 256, size=enc_o_sizes[3], v_coord=v_coord, stride=stride[3])
        self.conv_3_1 = ConvBlock(256, 256, size=enc_o_sizes[3], v_coord=v_coord, stride=stride[4])
        self.conv_3_2 = ConvBlock(256, 256, size=enc_o_sizes[3], v_coord=v_coord, stride=stride[5])
        self.conv_3_3 = ConvBlock(256, 512, dilation=1, pad=1, size=enc_o_sizes[6], v_coord=v_coord, stride=stride[6])
        self.conv_4_1 = ConvBlock(512, 512, dilation=2, pad=2, size=enc_o_sizes[6], v_coord=v_coord, stride=stride[7])
        self.conv_4_2 = ConvBlock(512, 512, dilation=2, pad=2, size=enc_o_sizes[6], v_coord=v_coord, stride=stride[8])
        self.conv_4_3 = ConvBlock(512, 512, dilation=2, pad=2, size=enc_o_sizes[6], v_coord=v_coord, stride=stride[9])

        self.d_conv_5_1 = FractionalConvolution(1024, 256, k_size=4, dilation=1, pad=1, stride=2, output_padding=0, v_coord=True, size=enc_o_sizes[3])
        self.d_conv_5_2 = ConvBlock(256, 256, dilation=1, pad=1, size=enc_o_sizes[3], v_coord=v_coord, stride=1)
        self.d_conv_5_3 = ConvBlock(256, 256, dilation=1, pad=1, size=enc_o_sizes[3], v_coord=v_coord, stride=1)
        self.d_conv_6_1 = FractionalConvolution(512, 128, k_size=4, dilation=1, pad=1, stride=2, output_padding=0, v_coord=True, size=enc_o_sizes[1])
        self.d_conv_6_2 = ConvBlock(128, 128, dilation=1, pad=1, size=enc_o_sizes[1], v_coord=v_coord, stride=1)
        self.d_conv_7_1 = FractionalConvolution(256, 64, k_size=4, dilation=1, pad=1, stride=2, output_padding=0, v_coord=True, size=enc_o_sizes[0])
        self.d_conv_7_2 = ConvBlock(64, 64, dilation=1, pad=1, size=enc_o_sizes[0], v_coord=v_coord, stride=1)
        self.d_conv_7_3 = ConvBlock(64, self.num_spheres*2, dilation=1, pad=1, size=enc_o_sizes[0],
                                    layer_norm=False, relu=False, v_coord=v_coord, stride=1)


    def forward(self, input_psv):
        device = input_psv.device
        b_size, _c, h, w = input_psv.shape
        e_v_coords_l = []
        for l in range(len(self.enc_o_factors)):
            coords = getattr(self, f'v_coord_{l}')
            coords = coords.repeat(b_size, 1, 1, 1).to(device)
            e_v_coords_l.append(coords)

        f0 = input_psv.view(b_size, self.inp_size, h, w)
        f_11 = self.conv_1_1(f0, e_v_coords_l[0])
        f_12 = self.conv_1_2(f_11, e_v_coords_l[1])
        f_21 = self.conv_2_1(f_12, e_v_coords_l[2])
        f_22 = self.conv_2_2(f_21, e_v_coords_l[3])
        f_31 = self.conv_3_1(f_22, e_v_coords_l[4])
        f_32 = self.conv_3_2(f_31, e_v_coords_l[5])
        f_33 = self.conv_3_3(f_32, e_v_coords_l[6])
        f_41 = self.conv_4_1(f_33, e_v_coords_l[7])
        f_42 = self.conv_4_2(f_41, e_v_coords_l[8])
        f_43 = self.conv_4_3(f_42, e_v_coords_l[9])
        d_51 = self.d_conv_5_1(torch.cat([f_43, f_33], dim=1), e_v_coords_l[9])
        d_52 = self.d_conv_5_2(d_51, e_v_coords_l[4])
        d_53 = self.d_conv_5_3(d_52, e_v_coords_l[4])
        d_61 = self.d_conv_6_1(torch.cat([d_53, f_22], dim=1), e_v_coords_l[4])
        d_62 = self.d_conv_6_2(d_61, e_v_coords_l[2])
        d_71 = self.d_conv_7_1(torch.cat([d_62, f_12], dim=1), e_v_coords_l[2])
        d_72 = self.d_conv_7_2(d_71, e_v_coords_l[0])
        d_73 = self.d_conv_7_3(d_72, e_v_coords_l[0])
        alpha = torch.sigmoid(d_73[:, :self.num_spheres, ...])
        blending_weight = torch.sigmoid(d_73[:, self.num_spheres:, ...])

        return alpha, blending_weight

    def _get_mul_scale_v_coords(self, v_coords, enc=True):
        v_list = []
        for f in self.enc_i_factors:
            v_list.append(F.interpolate(v_coords, scale_factor=(1/f), mode='bilinear', align_corners=False))
        return v_list
