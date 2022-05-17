# -*- coding: utf-8 -*-
# @Author : youngx
# @Time : 14:25  2022-05-11

import torch
import math
import torch.nn as nn
from .swinEncode import SwinTransformer
from collections import OrderedDict


class SceneRelation(nn.Module):
    def __init__(self, in_channels=256, channel_list=(256, 256, 256, 256),
                 out_channels=256, scale_aware_proj=True):
        super(SceneRelation, self).__init__()
        self.scale_aware_proj = scale_aware_proj

        if scale_aware_proj:
            self.scene_encoder = nn.ModuleList(
                [nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.ReLU(True),
                    nn.Conv2d(out_channels, out_channels, 1),
                ) for _ in range(len(channel_list))]
            )
        else:
            # 2mlp
            self.scene_encoder = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 1),
            )
        self.content_encoders = nn.ModuleList()
        self.feature_reencoders = nn.ModuleList()
        for c in channel_list:
            self.content_encoders.append(nn.Sequential(
                nn.Conv2d(c, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True))
            )
            self.feature_reencoders.append(
                nn.Sequential(
                    nn.Conv2d(c, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True))
            )

        self.normalizer = nn.Sigmoid()

    def forward(self, scene_feature, features: list):
        content_feats = [c_en(p_feat) for c_en, p_feat in zip(self.content_encoders, features)]
        if self.scale_aware_proj:
            scene_feats = [op(scene_feature) for op in self.scene_encoder]
            relations = [self.normalizer((sf * cf).sum(dim=1, keepdim=True)) for sf, cf in
                         zip(scene_feats, content_feats)]
        else:
            scene_feat = self.scene_encoder(scene_feature)
            relations = [self.normalizer((scene_feat * cf).sum(dim=1, keepdim=True)) for cf in content_feats]

        p_feats = [op(p_feat) for op, p_feat in zip(self.feature_reencoders, features)]

        refined_feats = [r * p for r, p in zip(relations, p_feats)]

        return refined_feats


class AssymetricDecoder(nn.Module):
    def __init__(self, in_channels=256, out_channels=128,
                 in_feat_output_strides=(4, 8, 16, 32),
                 out_feat_output_stride=4,
                 norm_fn=nn.BatchNorm2d,
                 num_groups_gn=None):
        super(AssymetricDecoder, self).__init__()
        if norm_fn == nn.BatchNorm2d:
            norm_fn_args = dict(num_features=out_channels)
        elif norm_fn == nn.GroupNorm:
            if num_groups_gn is None:
                raise ValueError('When norm_fn is nn.GroupNorm, num_groups_gn is needed.')
            norm_fn_args = dict(num_groups=num_groups_gn, num_channels=out_channels)
        else:
            raise ValueError('Type of {} is not support.'.format(type(norm_fn)))
        self.blocks = nn.ModuleList()
        for in_feat_os in in_feat_output_strides:
            num_upsample = int(math.log2(int(in_feat_os))) - int(math.log2(int(out_feat_output_stride)))

            num_layers = num_upsample if num_upsample != 0 else 1

            self.blocks.append(nn.Sequential(*[
                nn.Sequential(
                    nn.Conv2d(in_channels if idx == 0 else out_channels, out_channels, 3, 1, 1, bias=False),
                    norm_fn(**norm_fn_args) if norm_fn is not None else nn.Identity(),
                    nn.ReLU(inplace=True),
                    nn.UpsamplingBilinear2d(scale_factor=2) if num_upsample != 0 else nn.Identity(),
                )
                for idx in range(num_layers)]))

    def forward(self, feat_list: list):
        inner_feat_list = []
        for idx, block in enumerate(self.blocks):
            decoder_feat = block(feat_list[idx])
            inner_feat_list.append(decoder_feat)

        out_feat = sum(inner_feat_list) / 4.
        return out_feat


class FPN(nn.Module):
    def __init__(self, in_channels=256):
        super(FPN, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.c5 =self.make_conv(in_channels, [in_channels, in_channels * 2])
        self.c4 = self.make_conv(in_channels * 2, [in_channels, in_channels * 2])
        self.c3 = self.make_conv(in_channels * 2, [in_channels, in_channels * 2])
        self.c2 = self.make_conv(in_channels * 2, [in_channels, in_channels * 2])

    def make_conv(self, in_filters, filters_list):
        def conv2d(filter_in, filter_out, kernel_size, stride=1):
            pad = (kernel_size - 1) // 2 if kernel_size else 0
            return nn.Sequential(OrderedDict([
                ("conv",  nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
                ("bn", nn.BatchNorm2d(filter_out)),
                ("relu", nn.LeakyReLU(0.1)),
            ]))

        m = nn.Sequential(
            conv2d(in_filters, filters_list[0], 1),
            conv2d(filters_list[0], filters_list[1], 3),
            conv2d(filters_list[1], filters_list[0], 1),
            conv2d(filters_list[0], filters_list[1], 3),
            conv2d(filters_list[1], filters_list[0], 1),
        )
        return m

    def forward(self, features: list):
        c5, c4, c3, c2 = features[::-1]
        c5 = self.c5(c5)
        c4 = self.c4(torch.cat([self.upsample(c5), c4], dim=1))
        c3 = self.c3(torch.cat([self.upsample(c4), c3], dim=1))
        c2 = self.c2(torch.cat([self.upsample(c3), c2], dim=1))
        return [c2, c3, c4, c5]


class FarSeg(nn.Module):
    def __init__(self, image_size, min_channels=256, num_classes=1):
        super(FarSeg, self).__init__()
        self.backbone = SwinTransformer(image_size=image_size, patch_size=4, channels=3, dim=192)
        self.files = self.backbone.encode_channels
        self.conv = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(self.files[i], min_channels, 1),
                nn.ReLU(True)
            ) for i in range(len(self.files))]
        )
        self.fpn = FPN(in_channels=min_channels)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.scene_relation = SceneRelation()

        self.decoder = AssymetricDecoder(out_channels=min_channels)

        self.cls_pred_conv = nn.Conv2d(min_channels, num_classes, 1)
        self.upsample4x_op = nn.UpsamplingBilinear2d(scale_factor=4)

    def forward(self, x):
        feat_list = self.backbone(x)
        feat_list = [layer(feat) for layer, feat in zip(self.conv, feat_list)]
        fpn_feat_list = self.fpn(feat_list)

        # scene_relation
        c5 = feat_list[-1]
        c6 = self.gap(c5)
        refined_fpn_feat_list = self.scene_relation(c6, fpn_feat_list)

        del feat_list, fpn_feat_list
        torch.cuda.empty_cache()

        final_feat = self.decoder(refined_fpn_feat_list)
        cls_pred = self.cls_pred_conv(final_feat)
        cls_pred = self.upsample4x_op(cls_pred)

        del refined_fpn_feat_list
        torch.cuda.empty_cache()
        return cls_pred


if __name__ == '__main__':
    net = FarSeg(image_size=640)

    img = torch.randn(2, 3, 640, 640)
    out = net(img)
    print(out.shape)
