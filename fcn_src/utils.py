import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

# Extracts bias and non-bias parameters from a model.
def get_parameters(model, bias=False):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None


# from https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


# Convert to Long Tensor from Byte Tensor
class toLongTensor:
    def __call__(self, img):
        output = torch.from_numpy(np.array(img).astype(np.int32)).long()
        output[output == 255] = 21
        return output


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class):
    """ Returns overall accuracy and mean IoU """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iou = np.nanmean(iou)
    return acc, mean_iou


class Colorize(object):
    """ Colorize the segmentation labels """
    def __init__(self, n=35, cmap=None):
        if cmap is None:
            raise NotImplementedError()
            self.cmap = labelcolormap(n)
        else:
            self.cmap = cmap
        self.cmap = self.cmap[:n]
    def preprocess(self, x):
        if len(x.size()) > 3 and x.size(1) > 1:
            # if x has a shape of [B, C, H, W],
            # where B and C denote a batch size and the number of semantic classe
            # then translate it into a shape of [B, 1, H, W]
            x = x.argmax(dim=1, keepdim=True).float()
        assert (len(x.shape) == 4) and (x.size(1) == 1), 'x should have a shape of [B, 1, H, W]'
        return x
    def __call__(self, x):
        x = self.preprocess(x)
        if (x.dtype == torch.float) and (x.max() < 2):
            x = x.mul(255).long()
        color_images = []
        gray_image_shape = x.shape[1:]
        for gray_image in x:
            color_image = torch.ByteTensor(3, *gray_image_shape[1:]).fill_(0)
            for label, cmap in enumerate(self.cmap):
                mask = (label == gray_image[0]).cpu()
                color_image[0][mask] = cmap[0]
                color_image[1][mask] = cmap[1]
                color_image[2][mask] = cmap[2]
            color_images.append(color_image)
        color_images = torch.stack(color_images)
        return color_images


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])


def get_color_map():
    """ returns N color map """
    N=25
    color_map = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r, g, b = 0, 0, 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7-j))
            g = g ^ (np.uint8(str_id[-2]) << (7-j))
            b = b ^ (np.uint8(str_id[-3]) << (7-j))
            id = id >> 3
        color_map[i, 0] = r
        color_map[i, 1] = g
        color_map[i, 2] = b
    color_map = torch.from_numpy(color_map)
    return color_map


# Conditional Random Field for better segmentation
# Refer to https://github.com/lucasb-eyer/pydensecrf for details.
def dense_crf(img, output_probs):
    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    img = np.ascontiguousarray(img)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=1, compat=3)
    d.addPairwiseBilateral(sxy=67, srgb=3, rgbim=img, compat=4)

    Q = d.inference(10)
    Q = np.array(Q).reshape((c, h, w))
    return Q


# Zero-pad image(or any array-like object) to 500x500.
def add_padding(img):
    w, h = img.shape[-2], img.shape[-1]
    MAX_SIZE = 500
    IGNORE_IDX = 21

    assert max(w, h) <= MAX_SIZE, f'both height and width should be less than {MAX_SIZE}'

    _pad_left = (MAX_SIZE - w) // 2
    _pad_right = (MAX_SIZE - w + 1) // 2
    _pad_up = (MAX_SIZE - h) // 2
    _pad_down = (MAX_SIZE - h + 1) // 2

    _pad = (_pad_up, _pad_down, _pad_left, _pad_right)

    padding_img = transforms.Pad(_pad)
    padding_target = transforms.Pad(_pad, fill=IGNORE_IDX)

    img = F.pad(img, pad=_pad)
    return img