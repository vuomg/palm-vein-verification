# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

class ArcMarginProduct(nn.Module):
    r"""
    Implement of Additive Angular Margin Loss (ArcFace)
    Paper: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    Args:
        in_features: size of each input sample (embedding size, e.g., 512)
        out_features: size of each output sample (number of classes / IDs)
        s: norm of input feature (scale factor, default: 30.0)
        m: margin (default: 0.50)
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        
        # Trọng số W của lớp Loss (Centers), kích thước [Class_Size, Embedding_Size]
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # 1. Cosine similarity: cos(theta) = (Input_norm * Weight_norm)
        # Input và Weight đều được chuẩn hóa L2 trước khi nhân tích vô hướng
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        # 2. Tính sin(theta)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # 3. Tính cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # Xử lý các trường hợp ngoại lệ (easy_margin) để training ổn định hơn
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
        # 4. Chuyển label thành one-hot encoding
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # 5. Cộng Margin vào đúng vị trí lớp ground-truth (target class)
        # Output = s * (cos(theta + m)) với lớp đúng, và s * cos(theta) với lớp sai
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output


class AdaCos(nn.Module):
    """
    AdaCos (Phiên bản từ model_pytorch.py)
    - Scale cố định (Fixed Scale) thay vì Dynamic để ổn định.
    - Trả về LOGITS (đã nhân s), KHÔNG trả về Softmax.
    - Dùng với nn.CrossEntropyLoss() bên ngoài.
    """
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
        
        # Trả về logits đã scale. 
        # Lưu ý: Bên ngoài file train phải dùng CrossEntropyLoss (tự có Softmax)
        return logits * self.s