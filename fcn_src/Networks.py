import torch.nn as nn
from torchvision.models.vgg import VGG, vgg16, make_layers

from .utils import *

cfg = {'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}

class FCN32(VGG):
    def __init__(self):
        super(FCN32, self).__init__(make_layers(cfg['vgg16']))
        self.numclass = 21
        
        # Network Layers 
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout2d()
        
        # fc layers in vgg are all converted into conv layers.
        self.fc6 = nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=7)
        self.fc7 = nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1)

        # prediction layers with 1x1 convolution layers.       
        self.linclass = nn.Conv2d(in_channels=4096, out_channels=self.numclass, kernel_size=1) 
        
        # Learnable upsampling layers in FCN model.
        self.upsample = nn.ConvTranspose2d(in_channels=self.numclass, out_channels=self.numclass, kernel_size=64, stride=32, bias=False)

        self._initialize_weights()

    def load_pretrained(self, pretrained_model):
        self.features = pretrained_model.features
        fc6 = pretrained_model.classifier[0]
        fc7 = pretrained_model.classifier[3]
        # Load pretrained weights from VGG. Reshape it to fit into conv layer. 
        self.fc6.load_state_dict({"weight": fc6.state_dict()['weight'].view(4096, 512, 7, 7), "bias": fc6.state_dict()['bias']})
        self.fc7.load_state_dict({"weight": fc7.state_dict()['weight'].view(4096, 4096, 1, 1), "bias": fc7.state_dict()['bias']})


    def vgg_layer_forward(self, x, indices):
        output = x
        start_idx, end_idx = indices
        for idx in range(start_idx, end_idx):
            output = self.features[idx](output)
        return output

    def vgg_forward(self, x):
        out = {}
        layer_indices = [0, 5, 10, 17, 24, 31]
        for layer_num in range(len(layer_indices)-1):
            x = self.vgg_layer_forward(x, layer_indices[layer_num:layer_num+2])
            out[f'pool{layer_num+1}'] = x
        return out
        
    def forward(self, x):
        from torchvision.transforms.functional import crop
        # Padding for aligning to the input size
        padded_x = F.pad(x, [100, 100, 100, 100], "constant", 0)
        vgg_features = self.vgg_forward(padded_x)
        vgg_pool5 = vgg_features['pool5'].detach()
        vgg_pool4 = vgg_features['pool4'].detach()
        vgg_pool3 = vgg_features['pool3'].detach()

        out = self.dropout(self.relu(self.fc6(vgg_pool5)))
        out = self.dropout(self.relu(self.fc7(out)))
        out = self.linclass(out)
        out = self.upsample(out)
        _, __, h, w = x.shape
        out = crop(out, 9, 9, h, w)
        return out

    # initialize transdeconv layer with bilinear upsampling.
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

                
class FCN8(FCN32):
    def __init__(self):
        super(FCN8, self).__init__()

        self.numclass = 21
        
        # Network Layers
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout2d()

        # fc layers in vgg are all converted into conv layers.
        self.fc6 = nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=7)
        self.fc7 = nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1)

        # prediction layers with 1x1 convolution layers.
        self.lc1 = nn.Conv2d(in_channels=4096, out_channels=self.numclass, kernel_size=1)
        self.lc2 = nn.Conv2d(in_channels=512, out_channels=self.numclass, kernel_size=1)
        self.lc3 = nn.Conv2d(in_channels=256, out_channels=self.numclass, kernel_size=1)

        # Learnable upsampling layers in FCN model.
        self.us1 = nn.ConvTranspose2d(in_channels=self.numclass, out_channels=self.numclass, kernel_size=4, stride=2, bias=False)
        self.us2 = nn.ConvTranspose2d(in_channels=self.numclass, out_channels=self.numclass, kernel_size=4, stride=2, bias=False)
        self.us3 = nn.ConvTranspose2d(in_channels=self.numclass, out_channels=self.numclass, kernel_size=16, stride=8, bias=False)

        # initialize deconv layer with bilinear upsampling.
        self._initialize_weights()

    def forward(self, x):
        # Todo: 여기 Inherit 한 형태로 적을 수 있을까?
        from torchvision.transforms.functional import crop
        _, __, H, W = x.shape
        
        # Padding for aligning to the input size
        padded_x = F.pad(x, [100, 100, 100, 100], "constant", 0)
        vgg_features = self.vgg_forward(padded_x)
        vgg_pool5 = vgg_features['pool5'].detach()
        vgg_pool4 = vgg_features['pool4'].detach()
        vgg_pool3 = vgg_features['pool3'].detach()

        out = self.dropout(self.relu(self.fc6(vgg_pool5)))
        out = self.dropout(self.relu(self.fc7(out)))
        out = self.lc1(out)
        out = self.us1(out)
        _, __, h, w = out.shape
        out = out + crop(self.lc2(vgg_pool4 * 0.01), 5, 5, h, w)
        out = self.us2(out)

        _, __, h, w = out.shape
        out = out + crop(self.lc3(vgg_pool3 * 0.0001), 9, 9, h, w)
        out = self.us3(out)

        out = crop(out, 31, 31, H, W)

        return out