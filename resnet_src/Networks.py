from .Blocks import *


class ExampleNetwork(nn.Module):
    def __init__(self, nf, block_type='mlp'):
        super(ExampleNetwork, self).__init__()
        if block_type == 'mlp':
            block = MLPBlock
            # Since shape of input image is 3 x 32 x 32, the size of flattened input is 3*32*32. 
            self.mlp = block(3*32*32, nf)
            self.fc = nn.Linear(nf, 10)
        else:
            raise Exception(f"Wrong type of block: {block_type}.Expected : mlp")

    def forward(self, x):
        output = self.mlp(x.view(x.size()[0], -1))
        output = self.fc(output)
        return output


class ResNet(nn.Module):
    def __init__(self, nf, block_type='conv', num_blocks=[1, 1, 1]):
        super(ResNet, self).__init__()        
        self.block_type = block_type
        
        # Define blocks according to block_type
        if self.block_type == 'conv':
            block = ConvBlock
            block_args = lambda x: (x, x, 3, 1, 1)
        elif self.block_type == 'resPlain':
            block = ResBlockPlain
            block_args = lambda x: (x,)
        elif self.block_type == 'resBottleneck':
            block = ResBlockBottleneck
            block_args = lambda x: (x, x//2)
        elif self.block_type == 'inception':
            block = InceptionBlock
            block_args = lambda x: (x, x)
        else:
            raise Exception(f"Wrong type of block: {block_type}")
            
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv1 = nn.Conv2d(3, nf*1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nf*1)
        self.block1 = nn.Sequential(*[block(*block_args(nf)) for _ in range(num_blocks[0])])
        
        self.conv2 = nn.Conv2d(nf*1, nf*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(nf*2)
        self.block2 = nn.Sequential(*[block(*block_args(nf*2)) for _ in range(num_blocks[1])])
        
        self.conv3 = nn.Conv2d(nf*2, nf*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(nf*4)
        self.block3 = nn.Sequential(*[block(*block_args(nf*4)) for _ in range(num_blocks[2])])


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(nf*4, 10)

    def forward(self, x):
        output = self.block1(self.maxpool(self.relu(self.bn1(self.conv1(x)))))
        output = self.block2(self.maxpool(self.relu(self.bn2(self.conv2(output)))))
        output = self.block3(self.maxpool(self.relu(self.bn3(self.conv3(output)))))
        output = self.linear(self.flatten(self.avgpool(output)))
        return output