import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Các Block cơ bản (Giữ nguyên như bạn đã viết) ---
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bias=False, activation='relu'):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=use_bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'relu6':
            x = F.relu6(x)
        return x

class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bias=False, activation='relu'):
        super(SeparableConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=use_bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=use_bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'relu6':
            x = F.relu6(x)
        return x

class BottleneckV2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, t, stride, activation='relu'):
        super(BottleneckV2, self).__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * t))
        
        layers = []
        if t != 1:
            layers.append(ConvBlock(in_channels, hidden_dim, 1, 1, 0, activation=activation))
        
        padding = (kernel_size - 1) // 2
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'relu6':
            layers.append(nn.ReLU6(inplace=True))
            
        layers.append(nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, c, ks, t, s, n, activation='relu'):
        super(InvertedResidualBlock, self).__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(BottleneckV2(in_channels, c, ks, t, s, activation=activation))
        for _ in range(1, n):
            self.blocks.append(BottleneckV2(c, c, ks, t, 1, activation=activation))
            
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_list=[1, 2, 4]):
        super(SpatialPyramidPooling, self).__init__()
        self.pool_list = pool_list
        
    def forward(self, x):
        num_batch = x.size(0)
        outputs = []
        for size in self.pool_list:
            # ĐÃ FIX: Dùng Adaptive Max Pool giống Keras
            tensor = F.adaptive_max_pool2d(x, output_size=(size, size))
            outputs.append(tensor.view(num_batch, -1))
        return torch.cat(outputs, dim=1)

# --- BỔ SUNG CLASS ADACOS ---
class AdaCos(nn.Module):
    def __init__(self, num_features, num_classes, m=0.50):
        super(AdaCos, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = math.sqrt(2) * math.log(num_classes - 1)
        self.m = m
        self.W = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        # Normalize features & weights
        x = F.normalize(input)
        W = F.normalize(self.W)
        logits = F.linear(x, W)
        
        if label is None:
            return logits * self.s
        
        # Trong thực tế training, bạn nên dùng Dynamic s.
        # Ở đây dùng Fixed s để đảm bảo code chạy được ngay.
        return logits * self.s

# --- CLASS CHÍNH MPSNet ---
class MPSNet(nn.Module):
    def __init__(self, feature_dim=1024, input_channels=3, dropout=0.2): # Default input_channels=3
        super(MPSNet, self).__init__()
        
        self.c = [32, 32, 64, 64, 128]
        self.t = [1, 2, 2, 3, 2]
        self.s = [2, 2, 2, 2, 1]
        self.n = [1, 2, 2, 3, 2]
        
        activation = 'relu'
        
        self.m0_conv = ConvBlock(input_channels, self.c[0], 3, self.s[0], 1, activation=activation)
        
        self.block1_inv = InvertedResidualBlock(self.c[0], self.c[1], 3, self.t[1], self.s[1], self.n[1], activation=activation)
        self.block1_sep = SeparableConvBlock(self.c[0], self.c[1], 3, self.s[1], 1, activation=None)
        
        self.block2_inv = InvertedResidualBlock(self.c[1], self.c[2], 3, self.t[2], self.s[2], self.n[2], activation=activation)
        self.block2_sep = SeparableConvBlock(self.c[1], self.c[2], 3, self.s[2], 1, activation=None)
        
        self.block3_inv = InvertedResidualBlock(self.c[2], self.c[3], 3, self.t[3], self.s[3], self.n[3], activation=activation)
        self.block3_sep = SeparableConvBlock(self.c[2], self.c[3], 3, self.s[3], 1, activation=None)
        
        self.block4_inv = InvertedResidualBlock(self.c[3], self.c[4], 3, self.t[4], self.s[4], self.n[4], activation=activation)
        self.block4_sep = SeparableConvBlock(self.c[3], self.c[4], 3, self.s[4], 1, activation=None)
        
        # SPP Block
        self.spp_dropout = nn.Dropout(p=dropout)
        self.spp = SpatialPyramidPooling(pool_list=[1, 2, 4])
        
        spp_output_dim = 21 * self.c[4]
        self.spp_bn = nn.BatchNorm1d(spp_output_dim)
        
        # BỔ SUNG: Dropout thứ 2 trước Embedding (giống nets.py)
        self.emb_dropout = nn.Dropout(p=dropout)
        
        # Embedding Layer
        self.embedding = nn.Sequential(
            nn.Linear(spp_output_dim, feature_dim),
            nn.BatchNorm1d(feature_dim)
        )
        
    def forward(self, x):
        m0 = self.m0_conv(x)
        
        m1 = self.block1_inv(m0)
        m0_sep = self.block1_sep(m0)
        a1 = m1 + m0_sep
        
        m2 = self.block2_inv(a1)
        a1_sep = self.block2_sep(a1)
        a2 = m2 + a1_sep
        
        m3 = self.block3_inv(a2)
        a2_sep = self.block3_sep(a2)
        a3 = m3 + a2_sep
        
        m4 = self.block4_inv(a3)
        a3_sep = self.block4_sep(a3)
        a4 = m4 + a3_sep
        
        # SPP Logic
        x = self.spp_dropout(a4)
        x = self.spp(x)
        x = self.spp_bn(x)
        
        # Embedding Logic
        x = self.emb_dropout(x) # Chạy qua dropout thứ 2
        x = self.embedding(x)
        
        return x