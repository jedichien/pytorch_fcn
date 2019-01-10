import torch
import torch.nn as nn
from torch.nn.functional import relu
from torchvision.models import vgg16

base_model = vgg16(pretrained=True)

class FCN(nn.Module):
    def __init__(self, pretrained_net=base_model, n_class=2, skip_net_index=[4, 9, 16, 23, 30]):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = self._remove_last_from_pretrained(pretrained_net)
        self.skip_net_index = skip_net_index
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)
    
    def forward(self, x):
        _x = [] # 0: pool1, 1: pool2, 2: pool3, 3: pool4, 4: pool5
        for _i, _idx in enumerate(self.skip_net_index):
            _idx += 1
            if _i == 0:
                x = self.pretrained_net[:_idx](x)
            else:
                x = self.pretrained_net[self.skip_net_index[_i-1]+1:_idx](x)
            _x.append(x)
        # skip pool4
        sc = self.bn1(self.relu(self.deconv1(_x[4]))) # (N, 512, x.H/16, x.W/16)
        sc = sc + _x[3] # element-wise add pool4
        # skip (pool4 + pool3)
        sc = self.bn2(self.relu(self.deconv2(_x[3]))) # (N, 256, x.H/8, x.W/8)
        sc = sc + _x[2] # element-wise add pool3
        # skip (pool4 + pool3 + pool2)
        sc = self.bn3(self.relu(self.deconv3(_x[2]))) # (N, 128, x.H/4, x.W/4)
        sc = sc + _x[1] # element-wise add pool2
        # skip (pool4 + pool3 + pool2 + pool1)
        sc = self.bn4(self.relu(self.deconv4(_x[1]))) # (N, 64, x.H/2, x.W/2)
        sc = sc + _x[0] # element-wise add pool1
        # UPSAMPLING
        sc = self.bn5(self.relu(self.deconv5(sc))) # (N, 32, x.H, x.W)
        sc = self.classifier(sc)
        return sc
    
    def _remove_last_from_pretrained(self, model, freeze=True):
        _seq_no_last = next(model.children())
        if freeze:
            for param in _seq_no_last.parameters():
                param.requires_grad = False
        return _seq_no_last
