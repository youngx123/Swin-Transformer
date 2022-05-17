# -*- coding: utf-8 -*-
# @Author : youngx
# @Time : 14:26  2022-05-11


from loss import segmentation_loss
from decodeHead.decodeHead import FarSeg
import torch

if __name__ == '__main__':
    net = FarSeg(image_size=640, num_classes=2)

    img = torch.randn(2, 3, 640, 640)
    label = torch.ones((2, 640, 640))
    out = net(img)
    print(out.shape)
    loss = segmentation_loss(out, label, 1, AnnealingSoftmaxFocalloss=True)

