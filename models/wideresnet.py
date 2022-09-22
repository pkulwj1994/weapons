
from . import utils, layers, layerspp, normalization
from .utils import register_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from operator import __add__

conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
nin_0 = layerspp.NIN
get_act = layers.get_act
default_init = layers.default_init
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


def get_final_act(config):
  """Get activation functions from the config file."""

  if config.ebm.final_act.lower() == 'elu':
    return nn.ELU()
  elif config.ebm.final_act.lower() == 'relu':
    return nn.ReLU()
  elif config.ebm.final_act.lower() == 'lrelu':
    return nn.LeakyReLU(negative_slope=0.2)
  elif config.ebm.final_act.lower() == 'swish':
    return nn.SiLU()
  else:
    raise NotImplementedError('activation function does not exist!')


class normalize(nn.Module):
    def __init__(self, config):
        super(normalize, self).__init__()
        self.normalize = config.model.normalization
        if self.normalize is not None:
            self.norm = get_normalization(config)

    def forward(self, inputs):
        if self.normalize is not None:
            inputs = self.norm(inputs)
        return inputs



class linear(nn.Module):
    """
    fully-connected layer
    """
    def __init__(self, in_dim, out_dim, bias=True, init_scale=1., spec_norm=False):
        super(linear, self).__init__()

        if spec_norm:
            self.linear = nn.utils.spectral_norm(nn.Linear(in_dim, out_dim, bias))
        else:
            self.linear = nn.Linear(in_dim, out_dim, bias)
        self.linear.weight.data = default_init(init_scale)(self.linear.weight.data.shape)
        nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)


class conv2d(nn.Module):
    """
    2d convolutional layer
    """
    def __init__(self, in_dim, out_dim, filter_size=(3, 3), stride=1, dilation=None, pad='same', bias=True, init_scale=1.,
                spec_norm=False, use_scale=False):
        super(conv2d, self).__init__()

        if spec_norm:
            self.conv2d = nn.utils.spectral_norm(nn.Conv2d(in_dim, out_dim, filter_size, stride, pad, bias=bias))
        else:
            self.conv2d = nn.Conv2d(in_dim, out_dim, filter_size, stride, pad, bias=bias)
        if use_scale:
            self.g = nn.Parameter(torch.ones((out_dim, 1, 1)), requires_grad=True)
        self.use_scale = use_scale
        self.conv2d.weight.data = default_init(init_scale)(self.conv2d.weight.data.shape)
        nn.init.zeros_(self.conv2d.bias)

    def forward(self, inputs):
        # h = self.pad(inputs)
        h = self.conv2d(inputs)
        if self.use_scale:
            h = h * self.g
        return h


class nin(nn.Module):
    def __init__(self, in_dim, num_units, init_scale=0.1, spec_norm=False):
        super().__init__()
        if spec_norm:
            self.nin = nn.utils.spectral_norm(nin_0(in_dim, num_units, init_scale))
        else:
            self.nin = nin_0(in_dim, num_units, init_scale)

    def forward(self, input):
        return self.nin(input)


class resnet_block(nn.Module):
    def __init__(self, config, in_ch, out_ch=None, temb_dim=None, dropout=0.):
        super(resnet_block, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch if out_ch else in_ch
        self.act = get_act(config)
        self.conv_shortcut = config.ebm.res_conv_shortcut
        self.spec_norm = config.ebm.spec_norm
        self.use_scale = config.ebm.res_use_scale

        self.normalize_1 = normalize(config)
        self.normalize_2 = normalize(config)
        self.dropout_0 = nn.Dropout(dropout)
        if temb_dim is not None:
            self.dense = linear(in_dim=temb_dim, out_dim=self.out_ch, spec_norm=self.spec_norm)
        self.conv2d_1 = conv2d(in_dim=self.in_ch, out_dim=self.out_ch, spec_norm=self.spec_norm)
        self.conv2d_2 = conv2d(in_dim=self.out_ch, out_dim=self.out_ch, init_scale=0., spec_norm=self.spec_norm, use_scale=self.use_scale)
        if self.conv_shortcut:
            self.conv2d_shortcut = conv2d(in_dim=self.in_ch, out_dim=self.out_ch, spec_norm=self.spec_norm)
        else:
            self.nin_shortcut = nin(in_dim=self.in_ch, out_dim=self.out_ch, spec_norm=self.spec_norm)
        # print('{}: x={}'.format(self.name, input_shape))

    def forward(self, inputs, temb=None):
        B, C, H, W= inputs.shape
        x = inputs
        h = inputs

        h = self.act(self.normalize_1(h))
        h = self.conv2d_1(h)

        if temb is not None:
        # add in timestep embedding
            h += self.dense(self.act(temb))[:, :, None, None]
            
        h = self.act(self.normalize_2(h))
        h = self.dropout_0(h)
        h = self.conv2d_2(h)

        if C != self.out_ch:
            if self.conv_shortcut:
                x = self.conv2d_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        assert x.shape == h.shape
        return x + h


class downsample(nn.Module):
    def __init__(self, channels, with_conv=False, spec_norm=False):
        super(downsample, self).__init__()
        self.with_conv = with_conv

        if self.with_conv:
            self.conv2d = conv2d(in_dim=channels, out_dim=channels, filter_size=3, stride=2, spec_norm=spec_norm)
        # print('{}: x={}'.format(self.name, input_shape))

    def forward(self, inputs):
        B, C, H, W = inputs.shape
        if self.with_conv:
            x = F.pad(inputs, (0, 1, 0, 1))
            x = self.conv2d(inputs)
        else:
            x = F.avg_pool2d(inputs, kernel_size=2, stride=2, padding=0)

        return x


class attn_block(nn.Module):
    def __init__(self, config, channels):
        super(attn_block, self).__init__()
        self.normalize = normalize(config)
        self.nin_q = nin(channels, channels)
        self.nin_k = nin(channels, channels)
        self.nin_v = nin(channels, channels)

        self.nin_proj_out = nin(channels, channels, init_scale=0.)
        # print('{}: x={}'.format(self.name, input_shape))

    def forward(self, inputs):
        x = inputs
        B, C, H, W = x.shape

        h = self.normalize(x)
        q = self.nin_q(h)
        k = self.nin_k(h)
        v = self.nin_v(h)

        w = torch.einsum('bchw,bcHW->bhwHW', q, k) * (int(C) ** (-0.5))
        w = torch.reshape(w, [B, H, W, H * W])
        w = torch.nn.softmax(w, dim=-1)
        w = torch.reshape(w, [B, H, W, H, W])
        h = torch.einsum('bhwHW,bcHW->bchw', w, v)
        h = self.nin_proj_out(h)

        assert h.shape == x.shape

        return x + h


@register_model(name='ebm')
class WideResNet(nn.Module):
    def __init__(self, config):
        super(WideResNet, self).__init__()
        self.ch, self.ch_mult = config.model.nf, config.model.ch_mult
        self.num_res_blocks = config.model.num_res_blocks
        self.attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        self.num_resolutions = len(self.ch_mult)
        self.resamp_with_conv = config.ebm.resamp_with_conv
        self.use_attention = config.ebm.use_attention
        self.spec_norm = config.ebm.spec_norm
        self.act = get_act(config)
        self.final_act = get_final_act(config)
        channels = config.data.num_channels

        self.temb_linear_0 = linear(in_dim=self.ch, out_dim=self.ch * 4, spec_norm=self.spec_norm)
        self.temb_linear_1 = linear(in_dim=self.ch * 4, out_dim=self.ch * 4, spec_norm=self.spec_norm)
        self.temb_linear_2 = linear(in_dim=self.ch * 4, out_dim=self.ch * self.ch_mult[-1], spec_norm=False)

        self.res_levels = []
        self.attn_s = dict()
        self.downsample_s = []
        S = config.data.image_size

        # downsample
        # self.conv2d_in = conv2d(in_dim=channels, out_dim=self.ch, spec_norm=self.spec_norm)
        self.conv2d_in = conv3x3(channels, self.ch)
        in_ch = self.ch
        for i_level in range(self.num_resolutions):
            res_s = []
            if self.use_attention and S in self.attn_resolutions:
                self.attn_s[str(S)] = []
            for i_block in range(self.num_res_blocks):
                out_ch = self.ch * self.ch_mult[i_level]
                res_s.append(
                    resnet_block(
                        config=config, in_ch=in_ch, out_ch=out_ch, temb_dim=4*self.ch, dropout=dropout
                    )
                )
                in_ch = out_ch
                if self.use_attention and S in self.attn_resolutions:
                    self.attn_s[str(S)].append(attn_block(config, in_ch))
            res_s = nn.ModuleList(res_s)
            self.res_levels.append(res_s)

            if i_level != self.num_resolutions - 1:
                self.downsample_s.append(downsample(channels=in_ch, with_conv=self.resamp_with_conv, spec_norm=self.spec_norm))
                S = S // 2
        self.res_levels = nn.ModuleList(self.res_levels)
        # end
        self.normalize_out = normalize(config)
        
        # self.fc_out = linear(out_dim=1, spec_norm=False)

    def forward(self, x, t):
        B, _, _, S = x.shape
        assert x.dtype == torch.float32 and x.shape[2] == S
        if isinstance(t, int) or len(t.shape) == 0:
            t = torch.ones([B], dtype=torch.int32) * t

        # Timestep embedding
        temb = layers.get_timestep_embedding(t, self.ch)
        temb = self.temb_linear_0(temb)
        temb = self.temb_linear_1(self.act(temb))
        # downsample
        h = self.conv2d_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.res_levels[i_level][i_block](h, temb=temb)

                if self.use_attention:
                    if h.shape[-1] in self.attn_resolutions:
                        h = self.attn_s[str(h.shape[-1])][i_block](h)

            if i_level != self.num_resolutions - 1:
                h = self.downsample_s[i_level](h)
                
        # end
        h = self.final_act(h)
        h = torch.sum(h, (2, 3))
        temb_final = self.temb_linear_2(self.act(temb))
        h = torch.sum(h * temb_final, axis=1)
        return h