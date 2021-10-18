# Code citation for RegNet_v1
# Aaron Low (2020) RegNet source code (Version 1.0) [Source code]. https://github.com/aaronlws95/regnet
# Code citation for RegNet_v2
# Aaron Low (2020) RegNet source code (Version 2.0) [Source code]. https://github.com/aaronlws95/regnet
"""
Title: RegNet source code
Author: Aaron Low
Date: 2020
Code version: 1.0, 2.0
Availability: https://github.com/aaronlws95/regnet
"""
import torch
import torch.nn as nn
from torchvision import models
from src.fusion_net import fusion_module_C


def remove_layer(model, n):
    modules = list(model.children())[:-n]
    model = nn.Sequential(*modules)
    return model


def get_num_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):  # NIN block
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.1),
        nn.Conv2d(out_planes, out_planes, 1, 1, 0, bias=False),  # 1*1卷积
        nn.ReLU(inplace=True),
        nn.Conv2d(out_planes, out_planes, 1, 1, 0, bias=False),
        nn.ReLU(inplace=True)
    )


class RegNet_v3(nn.Module):
    def __init__(self):
        super(RegNet_v3, self).__init__()
        self.fusion = fusion_module_C(256, 256, 256)
        self.pool0 = nn.MaxPool2d(3, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(3, 1, 1)
        self.pool2 = nn.MaxPool2d(3, 1, 1)
        self.conv1 = nn.Sequential(nn.Conv2d(3, 96, 11, 4, 5, bias=True),
                                   nn.ReLU())
        self.conv1.apply(self.init_weights)
        self.RGB_net1 = nn.Sequential(
            nn.Conv2d(96, 96, 1, 1, 0, bias=True),
            nn.ReLU(),
            nn.Conv2d(96, 96, 1, 1, 0, bias=True),
            nn.ReLU(),
        )
        self.RGB_net1.apply(self.init_weights)
        self.RGB_net2 = nn.Sequential(
            nn.Conv2d(96, 192, 5, 1, 2, bias=True),
            nn.ReLU(),
            nn.Conv2d(192, 192, 1, 1, 0, bias=True),
            nn.ReLU(),
            nn.Conv2d(192, 192, 1, 1, 0, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.RGB_net2.apply(self.init_weights)
        self.RGB_net3 = nn.Sequential(
            nn.Conv2d(192, 256, 3, 1, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, 1, 1, 0, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, 1, 1, 0, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.RGB_net3.apply(self.init_weights)
        self.depth_net1 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(1, 48, 11, 4, 5, bias=True),
            nn.ReLU(),
            nn.Conv2d(48, 48, 1, 1, 0, bias=True),
            nn.ReLU(),
            nn.Conv2d(48, 48, 1, 1, 0, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.depth_net1.apply(self.init_weights)
        self.depth_net2 = nn.Sequential(
            nn.Conv2d(48, 128, 5, 1, 2, bias=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, 1, 1, 0, bias=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, 1, 1, 0, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.depth_net2.apply(self.init_weights)
        self.depth_net3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, 1, 1, 0, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, 1, 1, 0, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.depth_net3.apply(self.init_weights)
        self.depth_net = nn.Sequential(
            self.depth_net1,
            self.depth_net2,
            self.depth_net3
        )
        self.fuse1 = nn.Sequential(nn.Conv2d(768, 512, 5, 1, 2, bias=True),
                                   nn.ReLU(),
                                   nn.Conv2d(512, 512, 1, 1, 0, bias=True),
                                   nn.ReLU(),
                                   nn.Conv2d(512, 512, 1, 1, 0, bias=True),
                                   nn.ReLU(),
                                   )
        self.fuse1.apply(self.init_weights)
        self.fuse1_pool = nn.MaxPool2d(3, stride=2, padding=(0, 1))
        self.fuse2 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1, bias=True),
                                   nn.ReLU(),
                                   nn.Conv2d(512, 512, 1, 1, 0, bias=True),
                                   nn.ReLU(),
                                   nn.Conv2d(512, 512, 1, 1, 0, bias=True),
                                   nn.ReLU(),
                                   )

        self.fuse2.apply(self.init_weights)
        self.fuse2_pool = nn.MaxPool2d(3, stride=2)
        self.fc1 = nn.Linear(9216, 512)
        self.fc1.apply(self.init_weights)
        self.relu_fc1 = nn.ReLU()
        self.fc2_2_trans = nn.Linear(512, 256)
        self.fc2_2_rot = nn.Linear(512, 256)
        self.fc2_2_trans.apply(self.init_weights)
        self.fc2_2_rot.apply(self.init_weights)
        self.relu_fc2_trans = nn.ReLU()
        self.relu_fc2_rot = nn.ReLU()
        self.fc_final_trans = nn.Linear(256, 4)
        self.fc_final_rot = nn.Linear(256, 4)
        self.fc_final_trans.apply(self.init_weights)
        self.fc_final_rot.apply(self.init_weights)
        # learnable parameters
        self.sx = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.sx.data.fill_(1)
        self.sq = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.sq.data.fill_(1)

    def init_weights(self, layer):
        if type(layer) == nn.Conv2d:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
        elif type(layer) == nn.Linear:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, rgb_img, depth_img):
        # print("RegNet_v3_forward")
        # rgb_img.shape: torch.Size([2, 3, 352, 1216])
        # print("rgb_img.shape:", end=' ')
        # print(rgb_img.shape)

        rgb_features = self.conv1(rgb_img)
        # print("conv1", rgb_features.shape)
        rgb_features = self.RGB_net1(rgb_features)
        # print("RGB_net1", rgb_features.shape)
        rgb_features = self.pool0(rgb_features)
        # print("pool0", rgb_features.shape)
        rgb_features = self.RGB_net2(rgb_features)
        # print("RGB_net2", rgb_features.shape)
        rgb_features = self.RGB_net3(rgb_features)
        # print("rgb_features.shape", rgb_features.shape)
        depth_img = self.pool1(depth_img)
        depth_img = self.pool2(depth_img)
        depth_features = self.depth_net(depth_img)

        B, D, N, M = rgb_features.shape
        rgb_features = rgb_features.reshape(B, D, -1)
        B, D, N, M = depth_features.shape
        depth_features = depth_features.reshape(B, D, -1)

        concat_features = torch.cat((depth_features, rgb_features), 1)
        # print("concat_features.shape", concat_features.shape)
        matching_features = self.fusion(concat_features)
        # print("matching_features", matching_features.shape)
        matching_features = matching_features.reshape(B, -1, N, M)
        # print("matching_features", matching_features.shape)
        matching_features = self.fuse1(matching_features)
        # print("fuse1", matching_features.shape)
        matching_features = self.fuse1_pool(matching_features)
        # print("fuse1_pool", matching_features.shape)
        matching_features = self.fuse2(matching_features)
        # print("fuse2", matching_features.shape)
        matching_features = self.fuse2_pool(matching_features).squeeze()
        # print("fuse2_pool", matching_features.shape)
        matching_features = matching_features.reshape(-1, 9216)
        x = self.fc1(matching_features)
        x = self.relu_fc1(x)
        x_trans = self.fc2_2_trans(x)
        x_trans = self.relu_fc2_trans(x_trans)
        x_trans = self.fc_final_trans(x_trans)
        x_rot = self.fc2_2_rot(x)
        x_rot = self.relu_fc2_rot(x_rot)
        x_rot = self.fc_final_rot(x_rot)
        out = torch.cat((x_trans, x_rot), 1)
        return out, self.sx, self.sq


# Aaron Low (2020) RegNet source code (Version 2.0) [Source code]. https://github.com/aaronlws95/regnet
"""
Title: RegNet source code
Author: Aaron Low
Date: 2020
Code version: 2.0
Availability: https://github.com/aaronlws95/regnet
"""

"""
class RegNet_v2(nn.Module):
    def __init__(self):
        super(RegNet_v2, self).__init__()
        self.RGB_net = nn.Sequential(
            nn.Conv2d(3, 96, 11, 1, 0, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(96, 256, 5, 1, 0, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(256, 384, 3, 1, 0, bias=False),
        )
        self.RGB_net1 = nn.Sequential(
            nn.Conv2d(3, 96, 3, 1, 0, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(3),
        )
        self.RGB_net2 = nn.Sequential(
            nn.Conv2d(96, 256, 3, 1, 0, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(3),
        )
        self.RGB_net3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 0, bias=False),
        )

        self.RGB_conv1 = conv(3, 96, 3, 1, 0, 1)  # B*h*w*3->96
        self.RGB_conv2 = conv(96, 256, 3, 1, 0)   # B*h*w*256
        self.RGB_conv3 = conv(256, 384, 3, 1, 0)  # B*h*w*384

        self.depth_net = nn.Sequential(
            nn.Conv2d(1, 48, 11, 1, 0, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(48, 128, 5, 1, 0, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(128, 192, 3, 1, 0, bias=False),
        )

        self.matching = nn.Sequential(
            nn.Conv2d(576, 512, 3, 1, 0, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(512, 512, 3, 1, 0, bias=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 8)

    def forward(self, rgb_img, depth_img):
        # print("RegNet_v2_forward")

        # torch.Size([4, 96, 186, 619])
        # torch.Size([4, 256, 184, 617])
        # torch.Size([4, 384, 182, 615])

        # RF1 = self.RGB_conv1(rgb_img)  # RF1  # B*h*w*3->96
        # print(RF1.shape)
        # RF2 = self.RGB_conv2(RF1)  # RF2  # B*h*w*256
        # print(RF2.shape)
        # RF3 = self.RGB_conv3(RF2)  # RF3  # B*h*w*384
        # print(RF3.shape)

        # rgb_img.shape: torch.Size([4, 3, 188, 621])
        # torch.Size([4, 96, 62, 206])  # net1
        # torch.Size([4, 256, 20, 68])  # net2
        # torch.Size([4, 384, 18, 66])  # net3

        # print("rgb_img.shape:", end=' ')
        # print(rgb_img.shape)
        # rgb_features = self.RGB_net1(rgb_img)
        # print(rgb_features.shape, end='###net1\n')
        # rgb_features = self.RGB_net2(rgb_features)
        # print(rgb_features.shape, end='###net2\n')
        # rgb_features = self.RGB_net3(rgb_features)
        # print(rgb_features.shape, end='###net3\n')

        # 8192 /2 or /4

        rgb_features = self.RGB_net(rgb_img)
        # print(rgb_features.shape)
        depth_features = self.depth_net(depth_img)
        # print(depth_features.shape)
        concat_features = torch.cat((rgb_features, depth_features), 1)
        # print(concat_features.shape)
        matching_features = self.matching(concat_features).squeeze()
        x = self.fc1(matching_features)
        x = self.fc2(x)
        return x
"""

# Aaron Low (2020) RegNet source code (Version 1.0) [Source code]. https://github.com/aaronlws95/regnet
"""
Title: RegNet source code
Author: Aaron Low
Date: 2020
Code version: 1.0, 1.0
Availability: https://github.com/aaronlws95/regnet
"""

"""
class RegNet_v1(nn.Module):
    def __init__(self):
        super(RegNet_v1, self).__init__()
        self.RGB_net = remove_layer(models.resnet18(pretrained=True), 2)
        self.depth_net = remove_layer(models.resnet18(pretrained=False), 2)
        modules = list(self.depth_net.children())
        modules[0] = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.depth_net = nn.Sequential(*modules)
        for param in self.RGB_net.parameters():
            param.requires_grad = False
        self.matching = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 512, 1, 2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 8)

    def forward(self, rgb_img, depth_img):
        print("RegNet_v1_forward")
        print("rgb_img.shape", end=':')
        print(rgb_img.shape)
        print("depth_img.shape", end=':')
        print(depth_img.shape)
        rgb_features = self.RGB_net(rgb_img)
        print("rgb_features.shape", end=':')
        print(rgb_features.shape)
        depth_features = self.depth_net(depth_img)
        print("depth_features.shape", end=':')
        print(depth_features.shape)
        concat_features = torch.cat((rgb_features, depth_features), 1)
        print("concat_features.shape", end=':')
        print(concat_features.shape)
        matching_features = self.matching(concat_features).squeeze()
        x = self.fc1(matching_features)
        x = self.fc2(x)
        return x
"""
