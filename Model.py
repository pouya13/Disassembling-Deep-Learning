import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.relu = nn.SELU(inplace=True)
        self.conv2 = nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.downsample = downsample
        self.stride = stride
        
        ##################################################################
#         self.avg = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
#         self.fc1 = nn.Linear(out_channel, out_channel//2)
#         self.act1 = nn.SELU()
#         self.fc2 = nn.Linear(out_channel//2,out_channel)
#         self.act2 = nn.SELU()
        ##################################################################

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        #################################################
#         p1 = self.avg(out)
#         p1 = torch.flatten(p1, 1)
#         p1 = self.fc1(p1)
#         p1 = self.act1(p1)
#         p1 = self.fc2(p1)
#         p1 = self.act2(p1)
#         out = torch.einsum('ij, ijk -> ijk', p1, out)
        #################################################

        out += identity
        out = self.relu(out)

        return out

############################################################################

class MyResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super().__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv1d(50, self.inplanes, kernel_size=3, stride=2, padding=1,
                                bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.SELU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.MaxPool1d(1)
        self.fc = nn.Linear(512*8, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None  

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm1d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)           # 224x224
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)         # 112x112

        x = self.layer1(x)          # 56x56
        x = self.layer2(x)          # 28x28
        x = self.layer3(x)          # 14x14
        x = self.layer4(x)          # 7x7

        x = self.avgpool(x)         # 1x1
#         print(x.shape)
        x = torch.flatten(x, 1)     # convert 1 X 1 to vector
#         print(x.shape)
        x = self.fc(x)

        return x

layers=[3, 4, 6, 3]
model = MyResNet(BasicBlock, layers, num_classes=12)
