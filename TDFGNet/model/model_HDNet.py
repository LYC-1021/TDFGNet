import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
import os
from matplotlib import pyplot as plt
import torch.fft
import numpy as np
from models.layers import DTUM, DTUM_lyc


kernels_all = [[] for i in range(5)]
num_cycle1 = [1, 2, 3, 4, 5]  

kernels_all2 = [[] for i in range(7)]
num_cycle2 = [1, 2, 3, 4, 5, 6, 7] 

kernels_all3 = [[] for i in range(1)]

kernels_all4 = [[] for i in range(1)]

def GenerateKernels():
    """
    生成固定权值卷积核
    :return: None
    """
    for i in num_cycle1: 
        kernels = []
        for j in range(i):  
            k_size = (2 * i) + 1  
            kernel = np.zeros(shape=(k_size, k_size)).astype(np.float32)  
            lt_y = lt_x = k_size // 2 - ((j + 1) * 2 - 1) // 2 
            red_size = (j + 1) * 2 - 1
            red_val = 1 / kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size].size 
            kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size] = red_val 
            blue_val = -1 / (k_size ** 2 - kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size].size) 
            kernel[0:lt_x, :] = kernel[lt_x + red_size:, :] = kernel[:, :lt_y] = kernel[:, lt_y + red_size:] = blue_val 

            kernels.append(kernel)
        kernels_all[i - 1] = kernels
        pass
    return kernels_all

def GenerateKernels2():
    """
    生成固定权值卷积核
    :return: None
    """
    for i in num_cycle2:  
        kernels = []
        for j in range(1): 
            k_size = (2 * i) + 1 
            kernel = np.zeros(shape=(k_size, k_size)).astype(np.float32) 
            lt_y = lt_x = k_size // 2 - ((j + 1) * 2 - 1) // 2  
            red_size = (j + 1) * 2 - 1 
            red_val = 1 / kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size].size  
            kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size] = red_val  
            blue_val = -1 / (k_size ** 2 - kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size].size) 
            kernel[0:lt_x, :] = kernel[lt_x + red_size:, :] = kernel[:, :lt_y] = kernel[:, lt_y + red_size:] = blue_val  
            kernels.append(kernel)
        kernels_all2[i - 1] = kernels
        pass
    return kernels_all2

def GenerateKernels3():
    kernel = np.ones(shape=(3, 3)).astype(np.float32)
    kernel = kernel / 9.0
    kernels_all3[0].append(kernel)
    return kernels_all3

def GenerateKernels4():
    kernel = np.ones(shape=(3, 3)).astype(np.float32)
    kernel = kernel / 8.0 * -1
    kernel[1, 1] = 0
    kernels_all4[0].append(kernel)
    return kernels_all4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
kernels = GenerateKernels()
weights = [
            nn.Parameter(data = torch.FloatTensor(k).unsqueeze(0).unsqueeze(0), requires_grad=False).to(device)
            for ks in kernels for k in ks
        ]
kernels2 = GenerateKernels3()
weights2 = [
            nn.Parameter(data = torch.FloatTensor(k).unsqueeze(0).unsqueeze(0), requires_grad=False).to(device)
            for ks in kernels2 for k in ks
        ]
kernels3 = GenerateKernels4()
weights3 = [
            nn.Parameter(data = torch.FloatTensor(k).unsqueeze(0).unsqueeze(0), requires_grad=False).to(device)
            for ks in kernels3 for k in ks
        ]

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, max(1, in_planes // 16), 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(max(1, in_planes // 16), in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        # out = out + residual
        return out

class MAC(nn.Module):
    def __init__(self, inplanes, outplanes, one, two, three, scales = 4):
        super(MAC, self).__init__()
        if outplanes % scales != 0: 
            raise ValueError('Planes must be divisible by scales')
        self.weights = weights[:]
        self.weights2 = weights2
        self.weights3 = weights3
        self.scales = scales
        self.relu = nn.ReLU(inplace = True)
        self.spx = outplanes // scales
        self.inconv = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1, 1, 0),
            nn.BatchNorm2d(outplanes)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.spx, self.spx, one, 1, one // 2, groups = self.spx),
            nn.BatchNorm2d(self.spx),
        )
        self.conv1[0].weight.data = self.weights[one // 2 - 1].repeat(self.spx, 1, 1, 1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.spx, self.spx, two, 1, 2, groups = self.spx, dilation=2),
            nn.BatchNorm2d(self.spx),
        )
        self.conv2[0].weight.data = self.weights[two // 2 - 1].repeat(self.spx, 1, 1, 1)

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.spx, self.spx, three, 1, 1, groups = self.spx),
        )
        self.conv3[0].weight.data = self.weights2[0].repeat(self.spx, 1, 1, 1)

        self.conv4 = nn.Sequential(
            nn.Conv2d(self.spx, self.spx, three, 1, 2, groups = self.spx, dilation=2),
        )
        self.conv4[0].weight.data = self.weights3[0].repeat(self.spx, 1, 1, 1)
        
        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(self.spx)
        )
        self.outconv = nn.Sequential(
            nn.Conv2d(outplanes, outplanes, 3, 1, 1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )
        self.ca = ChannelAttention(outplanes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.inconv(x)
        inputt = x
        xs = torch.chunk(x, self.scales, 1)
        ys = []
        ys.append(xs[0])
        ys.append(self.relu(self.conv1(xs[1])))
        ys.append(self.relu(self.conv2(xs[2] + ys[1])))
        temp = xs[3] + ys[2]
        temp1 = self.conv5(self.conv3(temp) + self.conv4(temp))
        ys.append(self.relu(temp1))
        y = torch.cat(ys, 1)

        y = self.outconv(y)

        output = self.relu(y + inputt)
        return output

class DHPF(nn.Module):
    def __init__(self, energy):
        super(DHPF, self).__init__()
        self.energy = energy
    
    def _determine_cutoff_frequency(self, f_transform, target_ratio):
        total_energy = self._calculate_total_energy(f_transform)
        target_low_freq_energy = total_energy * target_ratio

        for cutoff_frequency in range(1, min(f_transform.shape[0], f_transform.shape[1]) // 2):
            low_freq_energy = self._calculate_low_freq_energy(f_transform, cutoff_frequency)
            if low_freq_energy >= target_low_freq_energy:
                return cutoff_frequency
        return 5 
    
    def _calculate_total_energy(self, f_transform):
        magnitude_spectrum = torch.abs(f_transform)
        total_energy = torch.sum(magnitude_spectrum ** 2)
        return total_energy
    
    def _calculate_low_freq_energy(self, f_transform, cutoff_frequency):
        magnitude_spectrum = torch.abs(f_transform)
        height, width = magnitude_spectrum.shape

        low_freq_energy = torch.sum(magnitude_spectrum[
            height // 2 - cutoff_frequency:height // 2 + cutoff_frequency,
            width // 2 - cutoff_frequency:width // 2 + cutoff_frequency
        ] ** 2)
    
        return low_freq_energy

    def forward(self, x):
        B, C, H, W = x.shape
        f = torch.fft.fft2(x)
        fshift = torch.fft.fftshift(f)
        crow, ccol = H // 2, W // 2
        for i in range(B):
            cutoff_frequency = self._determine_cutoff_frequency(fshift[i, 0], self.energy) 
            fshift[i, :, crow - cutoff_frequency:crow + cutoff_frequency, ccol - cutoff_frequency:ccol + cutoff_frequency] = 0
        ishift = torch.fft.ifftshift(fshift)
        ideal_high_pass = torch.abs(torch.fft.ifft2(ishift))
        return ideal_high_pass 

class HDNet(nn.Module):
    def __init__(self, input_channels, block=ResNet):
        super(HDNet, self).__init__()
        param_channels = [16, 32, 64, 128, 256]
        param_blocks = [2, 2, 2, 2]
        energy = [0.1, 0.2, 0.4, 0.8]

        self.pool = nn.MaxPool2d(2, 2)
        self.sigmoid = nn.Sigmoid()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.conv_init = nn.Conv2d(input_channels, param_channels[0], 1, 1)
        self.py_init = self._make_layer2(input_channels, 1, block)

        self.encoder_0 = self._make_layer(param_channels[0], param_channels[0], block)
        self.encoder_1 = self._make_layer(param_channels[0], param_channels[1], block, param_blocks[0])
        self.encoder_2 = self._make_layer(param_channels[1], param_channels[2], block, param_blocks[1])
        self.encoder_3 = self._make_layer(param_channels[2], param_channels[3], block, param_blocks[2])
     
        self.middle_layer = self._make_layer(param_channels[3], param_channels[4], block, param_blocks[3])
        
        self.decoder_3 = self._make_layer2(param_channels[3]+param_channels[4], param_channels[3], block, param_blocks[2])
        self.decoder_2 = self._make_layer2(param_channels[2]+param_channels[3], param_channels[2], block, param_blocks[1])
        self.decoder_1 = self._make_layer2(param_channels[1]+param_channels[2], param_channels[1], block, param_blocks[0])
        self.decoder_0 = self._make_layer2(param_channels[0]+param_channels[1], param_channels[0], block)

        self.py3 = DHPF(energy[3])
        self.py2 = DHPF(energy[2])
        self.py1 = DHPF(energy[1])
        self.py0 = DHPF(energy[0])

        self.output_0 = nn.Conv2d(param_channels[0], 1, 1)
        self.output_1 = nn.Conv2d(param_channels[1], 1, 1)
        self.output_2 = nn.Conv2d(param_channels[2], 1, 1)
        self.output_3 = nn.Conv2d(param_channels[3], 1, 1)

        self.final = nn.Conv2d(4, 32, 3, 1, 1)


    def _make_layer(self, in_channels, out_channels, block, block_num=1):
        layer = []        
        layer.append(MAC(in_channels, out_channels, 3, 3, 3))
        for _ in range(block_num-1):
            layer.append(block(out_channels, out_channels))
        return nn.Sequential(*layer)
    
    def _make_layer2(self, in_channels, out_channels, block, block_num = 1):
        layer= []
        layer.append(block(in_channels, out_channels))
        for _ in range(block_num-1):
            layer.append(block(out_channels, out_channels))
        return nn.Sequential(*layer)

    def forward(self, x, warm_flag):
        
        x_e0 = self.encoder_0(self.conv_init(x)) #
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))

        x_m = self.middle_layer(self.pool(x_e3))
        
        x_d3 = self.decoder_3(torch.cat([x_e3, self.up(x_m)], 1))
        x_d2 = self.decoder_2(torch.cat([x_e2, self.up(x_d3)], 1))
        x_d1 = self.decoder_1(torch.cat([x_e1, self.up(x_d2)], 1))
        x_d0 = self.decoder_0(torch.cat([x_e0, self.up(x_d1)], 1))
        
        mask0 = self.output_0(x_d0)
        mask1 = self.output_1(x_d1)
        mask2 = self.output_2(x_d2)
        mask3 = self.output_3(x_d3)
        
        if warm_flag:
            x_py_init = self.py_init(x)
            x_py_v3 = x_py_init * self.sigmoid(self.up_8(mask3)) + x_py_init 
            x_py_v3 = self.py3(x_py_v3)

            x_py_v2 = x_py_v3 * self.sigmoid(self.up_4(mask2)) + x_py_v3 
            x_py_v2 = self.py2(x_py_v2)

            x_py_v1 = x_py_v2 * self.sigmoid(self.up(mask1)) + x_py_v2 
            x_py_v1 = self.py1(x_py_v1)

            x_py_v0 = x_py_v1 * self.sigmoid(mask0) + x_py_v1 
            x_py_v0 = self.sigmoid(self.py0(x_py_v0))

            output = self.final(torch.cat([mask0, self.up(mask1), self.up_4(mask2), self.up_8(mask3)], dim=1))
            output = output * x_py_v0 + output
            return [mask0, mask1, mask2, mask3], output
    
        else:
            output = self.output_0(x_d0)
            output = output
            return [], output

class HDNet_DTUM(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(HDNet_DTUM, self).__init__()

        self.UNet = HDNet(input_channels)#这里的32实际上是输出通道数而非num_class
        self.DTUM = DTUM_lyc(32, num_classes, num_frames=5)       
        self.warm_flag=True
    def forward(self, X_In, Old_Feat, OldFlag):
        #print("old:"+str(Old_Feat.shape))
        #print("x_in:"+str(X_In.shape))
        #print("oldf:"+str(OldFlag))
        # Old_Feat = X_In.shape[2]
        # Old_Feat = X_In[:, :, -1, :, :]           
        # Old_Feat = self.UNet(Old_Feat)
        # Old_Feat = torch.unsqueeze(Old_Feat, 2)


        FrameNum = X_In.shape[2]##确定帧数
        Features = X_In[:, :, -1, :, :]     #提取图像 BCHW
        A,Features = self.UNet(Features,self.warm_flag)      #特征图
        Features = torch.unsqueeze(Features, 2) #重新变成BCTHW
        if OldFlag == 1:  # append current features based on Old Features, for iteration input
            Features = torch.cat([Old_Feat, Features], 2)

        elif OldFlag == 0 and FrameNum > 1:
            for i_fra in range(FrameNum - 1):
                x_t = X_In[:, :, -2 - i_fra, :, :]
                A,x_t = self.UNet(x_t,self.warm_flag)
                x_t = torch.unsqueeze(x_t, 2)
                Features = torch.cat([x_t, Features], 2)
        Old_Feat=Features[:, :, -(FrameNum -1):, :, :].detach()  
        X_Out = self.DTUM(Features)

        return X_Out,Old_Feat
    
import torch
from thop import profile
from models.model_HDNet import HDNet_DTUM  # 确保路径正确

# ----------------- 配置 -----------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
B = 1        # batch size
C = 1        # 输入通道
H, W = 256, 256  # 图像尺寸
num_classes = 1   # 输出类别
num_frames = 5    # 输入帧数，可改为 3,5,7

# ----------------- 构造模型 -----------------
model = HDNet_DTUM(input_channels=C, num_classes=num_classes).to(device)
model.eval()

# ----------------- 构造输入 -----------------
X_In = torch.randn(B, C, num_frames, H, W).to(device)

# 构造 Old_Feat，如果帧数=1可以设为 None
if num_frames > 1:
    # 假设 UNet 输出通道为 32，特征尺寸为 H//8, W//8（根据 UNet 下采样倍数修改）
    Old_Feat = torch.randn(B, 32, num_frames-1, H//8, W//8).to(device)
else:
    Old_Feat = None

OldFlag = 0  # 正常推理使用 0

# ----------------- 计算 FLOPs 和 Params -----------------
flops, params = profile(model, inputs=(X_In, Old_Feat, OldFlag))

print("-"*60)
print(f"帧数: {num_frames}")
print("FLOPs: {:.4f} G".format(flops / 1e9))
print("Params: {:.4f} M".format(params / 1e6))
print("-"*60)

