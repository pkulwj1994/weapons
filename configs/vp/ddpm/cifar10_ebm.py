# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Config file for reproducing the results of DDPM on cifar-10."""
import torch
from configs.default_cifar10_configs import get_default_configs


def get_config():
  config = get_default_configs()

  # training
  training = config.training
  training.batch_size = 128
  training.sde = 'vpsde'
  training.continuous = False
#   training.reduce_mean = True
  training.snapshot_freq = 10000

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'ancestral_sampling'
  sampling.corrector = 'none'
#   sampling.predictor = 'none'
#   sampling.corrector = 'ald'
#   sampling.n_steps_each = 5
#   sampling.snr = 0.176
  

  # evaluation
  evaluate = config.eval
  evaluate.enable_loss = False
  evaluate.enable_sampling = True
  evaluate.enable_bpd = False
  evaluate.begin_ckpt = 1
  evaluate.end_ckpt = 1

  # data
  data = config.data
  data.centered = True

  # model
  model = config.model
  model.name = 'ebm'
  model.scale_by_sigma = False
  model.ema_rate = 0.9999
  model.normalization = None
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 8
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True

  # optim
  optim = config.optim
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 1e-4
  optim.beta1 = 0.9
  optim.amsgrad = False
  optim.eps = 1e-8
  optim.warmup = 0

  # ebm
  ebm = config.ebm
  ebm.use_ebm = True
  ebm.final_act = 'swish'
  ebm.res_conv_shortcut = True
  ebm.spec_norm = True
  ebm.res_use_scale = True
  ebm.resamp_with_conv = False
  ebm.use_attention = False

  config.device = torch.device('cuda:0')
  config.device_ids = [1, 2, 3, 0]


  return config
