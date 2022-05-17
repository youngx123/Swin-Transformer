# -*- coding: utf-8 -*-
# @Author : youngx
# @Time : 9:40  2022-05-11
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import rearrange
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np


class PatchEmbed(nn.Module):
    def __init__(self, pathsize, dim, channels):
        super(PatchEmbed, self).__init__()
        self.patchsize = (pathsize, pathsize)
        self.chans = pathsize * pathsize * channels
        self.Unfold = nn.Sequential(
            nn.Unfold(kernel_size=self.patchsize, stride=self.patchsize),
            Rearrange('b dim lenth-> b lenth dim')
        )
        self.proj = nn.Linear(self.chans, dim)
        self.norm_layer = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.Unfold(x)
        x = self.proj(x)
        x = self.norm_layer(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super(PatchMerging, self).__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm_layer = nn.LayerNorm(4 * dim)

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        # assert L == H * W, "dimention error"

        x = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
        if H % 2 != 0 and W % 2 != 0:
            h_pad = H % 2
            w_pad = W % 2
            pad = (0, 0, h_pad, 0, w_pad, 0)
            x = F.pad(x, pad)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        if H % 2 != 0 and W % 2 != 0:
            x = x[:, 1:, 1:, :]
        x = rearrange(x, 'b h w c -> b (h w) c')  # B H/2*W/2 4*C
        x = self.norm_layer(x)
        x = self.reduction(x)
        return x


def window_partition(x, window_size):
    """
    :param x: (B, H, W, C)
    :param window_size: window size
    :return: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    hNum = H // window_size
    wNum = W // window_size

    # assert H == hNum * window_size and W == wNum * window_size, "dimention dont match"
    windows = rearrange(x, 'b (hNum wsz1) (wNum wsz2) c->(hNum wNum b) wsz1 wsz2 c',
                        hNum=hNum, wNum=wNum, wsz1=window_size, wsz2=window_size)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    :param windows: (num_windows*B, window_size, window_size, C)
    :param window_size:  Window size
    :param H: Height of image
    :param W: Width of image
    :return: (B, H, W, C)
    """
    hNum = H // window_size
    wNum = W // window_size

    x = rearrange(windows, '(b hNum wNum) wsize1 wsize2 c ''-> b (hNum wsize1) (wNum wsize2) c',
                  hNum=hNum, wNum=wNum, wsize1=window_size, wsize2=window_size)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qk_scale=None,
                 attn_drop=0., proj_drop=0., qkv_bias=True):

        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        assert int(dim % self.num_heads) == 0, '{}/{}= {}, the result Must be divisible'.format(dim, self.num_heads,
                                                                                                dim / self.num_heads)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # B, self.num_heads, N, C // self.num_heads]
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position = self.relative_position_index.view(-1)
        relative_position_bias = self.relative_position_bias_table[relative_position].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1],
            -1)  # Wh*Ww,Wh*Ww,nH

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            # # 将不感兴趣的mask区域填充值为-100，那么在softmax计算公式中的指数计算时，该项趋于0，即对计算结果无贡献。（用加法替代乘法降低计算量）
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)  # Linear
        x = self.proj_drop(x)  # drop out
        return x


class WMSA(nn.Module):
    def __init__(self, window_size, dim, num_heads=6, mlp_ratio=4,
                 drop=0, drop_path=0., qkv_bias=False):
        super(WMSA, self).__init__()
        self.window_size = window_size
        self.attn = WindowAttention(dim=dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x, H, W):
        shortcut = x.clone()

        x = rearrange(x, "b (H W) C-> b H W C", H=H, W=W)
        x = window_partition(x, self.window_size)
        x = rearrange(x, 'b h w c->b (h w) c')
        attn_windows = self.attn(x, None)
        attn_windows = rearrange(attn_windows, 'b (h w) c->b h w c', h=self.window_size, w=self.window_size)

        # res attention
        # # B H W C
        x = window_reverse(attn_windows, self.window_size, H, W)

        # # B H * W C
        x = rearrange(x, 'b h w c -> b (h w) c')
        x = shortcut + self.drop_path(x)

        # res mlp
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class ShiftWMSA(nn.Module):
    def __init__(self, window_size, input_resolution, dim, num_heads=6,
                 mlp_ratio=4, drop=0, drop_path=0., qkv_bias=False):
        super(ShiftWMSA, self).__init__()
        self.window_size = window_size
        self.H = input_resolution[0]
        self.W = input_resolution[1]
        self.shift_size = window_size // 2
        self.attn = WindowAttention(dim=dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

        H, W = self.H, self.W
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        self.H = Hp
        self.W = Wp

        img_mask = torch.zeros((1, Hp, Wp, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))

        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)

        # # 将不感兴趣的mask区域填充值为-100，那么在softmax计算公式中的指数计算时，该项趋于0，即对计算结果无贡献。(后续中用加法替代乘法降低计算量）
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        shortcut = x.clone()

        x = rearrange(x, "b (H W) C-> b H W C", H=self.H, W=self.W)

        # calculate
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        shifted_x = window_partition(shifted_x, self.window_size)
        shifted_x = rearrange(shifted_x, 'b h w c->b (h w) c')
        attn_windows = self.attn(shifted_x, mask=self.attn_mask)
        attn_windows = rearrange(attn_windows, 'b (h w) c->b h w c', h=self.window_size, w=self.window_size)

        # res attention
        # # B H W C
        x = window_reverse(attn_windows, self.window_size, self.H, self.W)

        # # B H * W C
        x = rearrange(x, 'b h w c -> b (h w) c')
        x = shortcut + self.drop_path(x)

        # res mlp
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    def __init__(self, input_res, dim, depth, num_head, mlp_ratio,
                 norm_layer, window_size=7, qkv_bias=False):
        super(BasicLayer, self).__init__()
        self.input_res = (input_res, input_res)
        self.window_size = window_size
        self.depth = depth
        self.norm1 = norm_layer(dim)

        self.wmsa = WMSA(window_size=window_size, dim=dim, num_heads=num_head,
                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias)
        self.Shiftwmsa = ShiftWMSA(window_size=window_size, input_resolution=self.input_res, dim=dim,
                                   num_heads=num_head, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias)

        self.downsample = PatchMerging(self.input_res, dim)

    def forward(self, x):
        H, W = self.input_res

        x = rearrange(x, "b (H W) C-> b H W C", H=H, W=W)

        # # padding
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_r, pad_l, pad_b, pad_t))
        H = H + pad_r
        W = W + pad_b

        x = rearrange(x, "b H W C-> b (H W) C", H=H, W=W)
        for _ in range(self.depth):
            B, L, C = x.shape
            # assert L == H * W, "input feature has wrong size"

            x = self.norm1(x)
            x = self.wmsa(x, H, W)
            x = self.Shiftwmsa(x)

        # # remove padding
        down_x = rearrange(x, 'b (h w) c-> b h w c', h=H, w=W)
        down_x = down_x[:, pad_r:, pad_b:, :]

        x = rearrange(down_x, 'b h w c-> b (h w) c')
        x = self.downsample(x)
        down_x = rearrange(down_x, 'b h w c-> b c h w')
        return x, down_x


class SwinTransformer(nn.Module):
    def __init__(self, image_size, patch_size, channels, dim, window_size=7,
                 depth=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], mlp_ratio=4.,
                 qkv_bias=True, norm_layer=nn.LayerNorm):
        super(SwinTransformer, self).__init__()
        self.image_size = image_size
        self.dim = dim
        self.patchembed = PatchEmbed(patch_size, dim, channels)
        self.num_layer = len(depth)
        self.encode_channels = []
        layers = []
        for i in range(self.num_layer):
            self.input_res = image_size // (patch_size * (2 ** i))
            depth_i = depth[i]
            num_headi = num_heads[i]
            self.encode_channels += [int(dim * 2 ** i)]
            layers.append(BasicLayer(input_res=self.input_res, dim=int(dim * 2 ** i), depth=depth_i,
                                     num_head=num_headi, mlp_ratio=mlp_ratio, norm_layer=norm_layer,
                                     window_size=window_size, qkv_bias=qkv_bias))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.patchembed(x)
        encode_x = []
        for layer in self.layers:
            x, down_x = layer(x)
            encode_x.append(down_x)
        return encode_x


if __name__ == '__main__':
    net = SwinTransformer(image_size=640, patch_size=4, channels=3, dim=192,
                          depth=[2, 4, 6, 2], num_heads=[3, 6, 12, 24])

    img = torch.randn(2, 3, 640, 640)
    out = net(img)
    for outi in out:
        print(outi.shape)
    print(net.encode_channels)
    net.eval()
    torch.onnx.export(net, img, "swin0.onnx", verbose=0, training=torch.onnx.TrainingMode.EVAL,
                      input_names=["input"], output_names=["outnode"], opset_version=11)

    '''
    class M(torch.nn.Module):
        def __init__(self, shifts, dims):
            super(M, self).__init__()
            self.shifts = shifts
            self.dims = dims

        def forward(self, x):
            return torch.roll(x, self.shifts, self.dims)


    net = M(-1, 1)
    img = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).view(3, 3)
    print(img)
    out = net(img)
    print(out.shape, "\n\n", out)
    torch.onnx.export(net, img, "swin.onnx", verbose=0, training=torch.onnx.TrainingMode.EVAL,
                      input_names=["input"], output_names=["outnode"], opset_version=11)

    import onnxruntime
    import onnx

    onxmodel = onnxruntime.InferenceSession("swin.onnx")
    img_np = img.numpy()
    out2 = onxmodel.run(None, {"input":img_np})
    print("#####################")
    print(out2[0])
    '''
