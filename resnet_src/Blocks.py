import torch.nn as nn

class MLPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512,128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, out_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        output = self.act(self.bn1(self.fc1(x)))
        output = self.act(self.bn2(self.fc2(output)))
        output = self.act(self.bn3(self.fc3(output)))
        return output
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        
    def forward(self, x):
        output = self.act(self.bn(self.conv(x)))
        return output
    
class ResBlockPlain(nn.Module):
    def __init__(self, in_channels):
        super(ResBlockPlain, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
    
    def forward(self, x):
        output = self.act(self.bn(self.conv(x)))
        output = self.act(self.bn2(self.conv2(output)) + x)
        return output
    
class ResBlockBottleneck(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(ResBlockBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU()
        
    def forward(self, x):
        output = self.act(self.bn1(self.conv1(x)))
        output = self.act(self.bn2(self.conv2(output)))
        output = self.act(x + self.bn3(self.conv3(output)))
        return output

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        self.conv1_0 = nn.Conv2d(in_channels, out_channels//4, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels//4)

        self.conv3_0 = nn.Conv2d(in_channels, out_channels//2, 1, 1, 0, bias=False)
        self.conv3_1 =  nn.Conv2d(out_channels//2, out_channels//2, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels//2)
        self.bn31 = nn.BatchNorm2d(out_channels//2)

        self.conv5_0 = nn.Conv2d(in_channels, out_channels//8, 1, 1, 0, bias=False)
        self.conv5_1 = nn.Conv2d(out_channels//8, out_channels//8, 5, 1, 2, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels//8)
        self.bn51 = nn.BatchNorm2d(out_channels//8)

        self.mp = nn.MaxPool2d(3, 1, padding=1)
        self.convmp = nn.Conv2d(in_channels, out_channels//8, 1, 1, 0, bias=False)
        self.bnmp = nn.BatchNorm2d(out_channels//8)

        self.act = nn.ReLU()

    def forward(self, x):
        output1x1branch = self.act(self.bn1(self.conv1_0(x)))
        output3x3branch = self.act(self.bn31(self.conv3_1(self.act(self.bn3(self.conv3_0(x))))))
        output5x5branch = self.act(self.bn51(self.conv5_1(self.act(self.bn5(self.conv5_0(x))))))
        outputmpbranch = self.act(self.bnmp(self.convmp(self.mp(x))))
        output = torch.cat((output1x1branch, output3x3branch, output5x5branch, outputmpbranch), dim=1)
        return output