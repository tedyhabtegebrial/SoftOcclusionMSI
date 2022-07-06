import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer

def get_pos_enc_size(configs):
    return 2*2*(configs.num_octaves) + 2

def get_output_size(configs):
    n, f = configs.num_spheres, configs.feats_per_layer
    k = configs.num_layers
    return n + f*k + n*k
    
def get_positional_encoding(l_num, configs):
    scale = 2**(configs.num_blocks - l_num - 1)
    h, w = configs.height//scale, configs.width//scale
    phi = torch.linspace(0, math.pi, h).view(1, 1, h, 1)
    phi = phi.expand(1,1,h,w)
    theta = torch.linspace(2*math.pi, 0, w).view(1, 1, 1, w)
    theta = theta.expand(1,1,h,w)
    
    if not configs.add_noise:
        phi_theta = torch.cat([phi, theta], dim=1)
        encodings = [phi_theta]
        for octave in range(configs.num_octaves):
            enc_ = torch.cat([torch.sin(phi_theta*(2**octave)), torch.cos(phi_theta*(2**octave))], dim=1)
            encodings.append(enc_)
        encodings = torch.cat(encodings, dim=1)
        return encodings
    else:
        uniform = torch.distributions.Uniform(
            -1*torch.ones(1, 1, h, w), torch.ones(1, 1, h, w))
        noise = math.pi*uniform.sample()
        phi_theta_noise = torch.cat([phi, theta, noise], dim=1)
        phi_theta = torch.cat([phi, theta], dim=1)
        encodings = [phi_theta_noise]
        for octave in range(configs.num_octaves):
            enc_ = torch.cat([torch.sin(phi_theta*(2**octave)),
                              torch.cos(phi_theta*(2**octave))], dim=1)
            encodings.append(enc_)
        encodings = torch.cat(encodings, dim=1)
        return encodings

class DecodeFeatures(nn.Module):
    def __init__(self, configs):
        super(DecodeFeatures, self).__init__()
        self.in_chans = configs.feats_per_layer
        if self.in_chans == 3:
            self.net = nn.Sequential()
        else:
            self.out_chans = 3
            chans = [int(i.item()) for i in torch.linspace(self.in_chans, self.out_chans, 4)]
            if self.in_chans==3:
                self.net = nn.Sequential()
            chans = [self.in_chans, self.in_chans, self.out_chans]
            self.net = nn.Sequential(nn.Conv2d(chans[0], chans[1], kernel_size=1, stride=1, padding=0),
                                    nn.LeakyReLU(negative_slope=0.4),
                                    nn.Conv2d(chans[1], chans[2], kernel_size=1, stride=1, padding=0)
                                    )
    def forward(self, features):
        if self.in_chans == 3:
            return features
        else:
            feats = self.net(features)
            return torch.tanh(feats)


class ConvBlock(nn.Module):
    def __init__(self, input_chans, out_chans, convs_per_block=4, last=False):
        super(ConvBlock, self).__init__()
        self.in_c = input_chans
        self.out_c = out_chans
        self.num_l = convs_per_block
        self.last = last
        inp_chans = list(torch.linspace(
            self.in_c, self.out_c, self.num_l)) + [self.out_c]
        inp_chans = [int(i) for i in inp_chans]
        conv_layers_1 = []
        for i in range(1, len(inp_chans)//2):
            layer = nn.Conv2d(inp_chans[i-1], inp_chans[i], kernel_size=1, stride=1, padding=0)
            conv_layers_1.append(layer)
        self.layers_1 = torch.nn.ModuleList(conv_layers_1)
        conv_layers_2 = []
        for i in range(len(inp_chans)//2, len(inp_chans)):
            layer = nn.Conv2d(inp_chans[i-1], inp_chans[i],
                                 kernel_size=1, stride=1, padding=0)
            conv_layers_2.append(layer)
        self.layers_2 = torch.nn.ModuleList(conv_layers_2)

    def resize_rgb(self, input_img, b, h, w):
        b = input_img.shape[0]
        rbg_resized = F.interpolate(input_img, size=(h, w), mode='bilinear', align_corners=False)
        rbg_resized = rbg_resized.expand(b, 3, h, w)
        return rbg_resized

    def forward(self, x):
        # if not input_img is None:
        h, w = x.shape[2:]
        # x = torch.cat([inputs, pos_encodings, self.resize_rgb(input_img, b, h, w)])
        for l in self.layers_1:
            x = F.relu(l(x))
        if not self.last:
            x = F.interpolate(x, size=(h*2, w*2), mode='bilinear', align_corners=False)
        for l in self.layers_2[:-1]:
            x = F.relu(l(x))
        x = self.layers_2[-1](x)
        if not self.last:
            x = F.relu(x)
        return x



class SOMSINetwork(torch.nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.in_chans, self.out_chans, self.last_chan_in = self.get_inp_out_chans()
        # Creat Multi Scale 1x1 Conv Blocks
        self.build_conv_blocks()
        # Positional Encodings for each scale
        self.register_pos_encodings()

    def register_pos_encodings(self):
        for l in range(self.configs.num_blocks):
            self.register_buffer(
                f'pos_encodings_{l}', get_positional_encoding(l, self.configs))

    def build_conv_blocks(self):
        blocks = []
        convs_per_block = self.configs.convs_per_block
        in_chans, out_chans = self.in_chans, self.out_chans
        for l in range(self.configs.num_blocks):
            if l==self.configs.num_blocks-1:
                blocks.append(
                    ConvBlock(in_chans[l], out_chans[l], convs_per_block, last=True))
            else:
                blocks.append(
                    ConvBlock(in_chans[l], out_chans[l], convs_per_block))
        self.blocks = torch.nn.ModuleList(blocks)
        self.out_block = ConvBlock(
            self.last_chan_in, self.get_output_size(), convs_per_block, last=True)


    def get_output_size(self):
        configs = self.configs
        n, f = configs.num_spheres, configs.feats_per_layer
        k = configs.num_layers
        num_coef = configs.num_basis
        return n*(k + 1) + k*(f*num_coef)
    
    def get_inp_out_chans(self):
        ''' computes the input and output channels of each block '''
        configs = self.configs
        num_blks = self.configs.num_blocks
        max_chans = configs.max_chans
        min_chans = max(1, self.configs.num_spheres//128)*(configs.min_chans)
        out_chans = list(map(int, torch.linspace(max_chans, min_chans, num_blks)))
        pos_enc_size = get_pos_enc_size(configs)
        in_chans = [pos_enc_size] + [i+pos_enc_size for i in out_chans[:-1]]
        last_chan_in = out_chans[-1] + pos_enc_size
        if self.configs.with_pixel_input:
            in_chans = [i+3 for i in in_chans]
            last_chan_in += 3
        return in_chans, out_chans, last_chan_in

    def forward(self, input_rgb):
        rgb_list = [input_rgb]
        for i in range(1,self.configs.num_blocks):
            h_i, w_i = rgb_list[-1].shape[2:]
            rgb = F.interpolate(rgb_list[-1], size=(h_i//2, w_i//2), mode='bilinear',
                                align_corners=False)
            rgb_list.append(rgb)
        rgb_list = list(reversed(rgb_list))
        prev_act = None
        for i,l in enumerate(self.blocks):
            b, _, h, w = rgb_list[i].shape
            pos_encodings_i = getattr(self, f'pos_encodings_{i}')
            p_size = pos_encodings_i.shape[1]
            pos_encoding = (pos_encodings_i).expand(b, p_size, h, w)
            if i==0:
                x = torch.cat([rgb_list[i], pos_encoding], dim=1)
                prev_act = l(x)
            else:
                x = torch.cat([prev_act, rgb_list[i], pos_encoding], dim=1)
                prev_act = l(x)
        x = self.out_block(
            torch.cat([prev_act, rgb_list[-1], pos_encoding], dim=1))
        return self.split_result(x)
    
    def split_result(self, x):
        configs = self.configs
        data = {}
        
        alpha_s = [0,configs.num_spheres]
        assoc_s = [alpha_s[1], alpha_s[1]+configs.num_spheres*configs.num_layers]
        coeff_s = [assoc_s[1], assoc_s[1]+configs.num_layers *
                    configs.feats_per_layer*(configs.num_basis)]
        assoc_1 = (x[:, assoc_s[0]:assoc_s[1], :, :])
        b, cc, hh, ww = assoc_1.shape
        assoc_2 = assoc_1.view(b, configs.num_spheres, configs.num_layers, hh, ww)
        assoc_2 = torch.softmax(assoc_2, dim=2).view(assoc_1.shape)
        data['assoc'] = assoc_2
        data['alpha'] = torch.sigmoid(x[:, alpha_s[0]:alpha_s[1], :, :])
        # reflection coefficients
        # OR, if number of basis function is 1, this will simply be color features
        data['coefs'] = x[:, coeff_s[0]:coeff_s[1], :, :]
        if configs.feats_per_layer==3:
            data['coefs'] = torch.tanh(data['coefs'])
        # else:
        #     if not configs.no_coeff_relu:
        #         data['coefs'] = torch.relu(data['coefs'])
        return data

if __name__ == '__main__':
    class Configs:
        def __init__(self):
            self.num_blocks = 7
            self.height, self.width = 512, 1024
            self.num_octaves = 6
            self.num_layers = 3
            self.num_basis = 4
            self.use_learned_assoc = True
            self.feats_per_layer = 4
            self.add_noise = False
            self.min_chans = 64
            self.max_chans = 256
            self.convs_per_block = 4
            self.num_spheres = 32
            self.with_pixel_input = True
    
    conf = Configs()
    net = SOMSINetwork(conf)
    rgb = torch.rand(2, 3, conf.height, conf.width)
    res = net(rgb)
    for r in res.keys():
        print(r, res[r].shape)
