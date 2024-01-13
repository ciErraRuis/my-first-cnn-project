import torch
from torchsummary import summary
import torchvision

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", DEVICE)

"""
build the convneXt_small model
"""

# Configs

config = {
    'batch_size': 128,
    'lr': 2e-3,
    'epochs': 100,
    'drop_rate': [0, 0.1, 0.2, 0.2],
    'stage_depth': [3, 3, 27, 3],
    'stage_dim': [96, 192, 384, 768],
    'eps': 1e-3,
}


# BackBone Block (input: in_channel)

class Block(torch.nn.Module):
    def __init__(self, dim, drop_rate, layer_scale=1e-6):
        super().__init__()
        self.dwconv = torch.nn.Conv2d(dim, dim, 7, padding = 3, groups = dim)
        self.layernorm = torch.nn.LayerNorm(dim, eps=config['eps'])
        self.pwconv1 = torch.nn.Linear(dim, dim * 4)
        self.gelu = torch.nn.GELU()
        self.pwconv2 = torch.nn.Linear(dim * 4, dim)
        self.gamma = torch.nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        self.drop = self.drop = torchvision.ops.StochasticDepth(drop_rate, mode='row')

    def forward(self, x):
        re = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.layernorm(x)
        x = self.pwconv1(x)
        x = self.gelu(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = self.drop(x)
        x = x + re

        return x


# ConvneXt architecture

class ConvneXt(torch.nn.Module):

    """
    number of block in every stage: [3, 3, 27, 3]
    dim of every block: [96, 192, 384, 768]
    7000 classes
    """

    def __init__(self, class_num=7000):
        super().__init__()

        dim = config['stage_dim']
        drop_rate = config['drop_rate']

        self.stem = torch.nn.Conv2d(3, dim[0], 4, stride=4)
        self.layernorm_0 = torch.nn.LayerNorm(dim[0], eps=config['eps'])

        # stage1
        Block_1 = Block(dim[0], drop_rate[0])
        self.stage_1 = torch.nn.Sequential(
            Block_1,
            Block_1,
            Block_1
        )

        self.layernorm_1 = torch.nn.LayerNorm(dim[0], eps=config['eps'])
        self.downsample_1 = torch.nn.Conv2d\
            (dim[0], dim[1], 2, stride=2)
        
        # stage2
        Block_2 = Block(dim[1], drop_rate[1])
        self.stage_2 = torch.nn.Sequential(
            Block_2,
            Block_2,
            Block_2
        )

        self.layernorm_2 = torch.nn.LayerNorm(dim[1], eps=config['eps'])
        self.downsample_2 = torch.nn.Conv2d\
            (dim[1], dim[2], 2, stride=2)

        # stage3
        Block_3 = Block(dim[2], drop_rate[2])
        self.stage_3 = torch.nn.Sequential(
            Block_3,
            Block_3,
            Block_3,
            Block_3,
            Block_3,
            Block_3,
            Block_3,
            Block_3,
            Block_3,
            Block_3,
            Block_3,
            Block_3,
            Block_3,
            Block_3,
            Block_3,
            Block_3,
            Block_3,
            Block_3,
            Block_3,
            Block_3,
            Block_3,
            Block_3,
            Block_3,
            Block_3,
            Block_3,
            Block_3,
            Block_3,
        )

        self.layernorm_3 = torch.nn.LayerNorm(dim[2], eps=config['eps'])
        self.downsample_3 = torch.nn.Conv2d(dim[2], dim[3], 2, stride=2)

        # stage4
        Block_4 = Block(dim[3], drop_rate[3])
        self.stage_4 = torch.nn.Sequential(
            Block_4,
            Block_4,
            Block_4
        )

        self.layernorm_4 = torch.nn.LayerNorm(dim[3], eps=config['eps'])

        # classification
        self.cls_layer = torch.nn.Linear(dim[3], class_num)
        self.apply(self._init_weights)

        #record cls_layer's shape to prepare for the arcface model
        self.embedding = dim[3]
        self.class_num = class_num
        
        # from the official implementation of convnext
    def _init_weights(self, m):
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            torch.nn.init.constant_(m.bias, 0)

    # turning return_feats to true is used in verification task
    def forward(self, x, return_feats=False):

        x = self.stem(x)
        x = x.permute(0, 2, 3, 1)
        x = self.layernorm_0(x)
        x = x.permute(0, 3, 1, 2)

        x = self.stage_1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.layernorm_1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.downsample_1(x)

        x = self.stage_2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.layernorm_2(x)
        x = x.permute(0, 3, 1, 2)
        x = self.downsample_2(x)

        x = self.stage_3(x)
        x = x.permute(0, 2, 3, 1)
        x = self.layernorm_3(x)
        x = x.permute(0, 3, 1, 2)
        x = self.downsample_3(x)

        x = self.stage_4(x)

        x = x.mean((2, 3))
        feats = self.layernorm_4(x)
        out = self.cls_layer(feats)

        if return_feats:
            return feats
        else:
            return out