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

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""

import torch
import numpy as np
from scipy import integrate
from models import utils as mutils
from sde_lib import VESDE, VPSDE


def get_div_fn(fn):
  """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

  def div_fn(x, t, eps):
    with torch.enable_grad():
      x.requires_grad_(True)
      fn_eps = torch.sum(fn(x, t) * eps)
      grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
    x.requires_grad_(False)
    return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

  return div_fn


def get_likelihood_fn(sde, inverse_scaler, hutchinson_type='Rademacher',
                      rtol=1e-5, atol=1e-5, method='RK45', eps=1e-5):
  """Create a function to compute the unbiased log-likelihood estimate of a given data point.

  Args:
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    inverse_scaler: The inverse data normalizer.
    hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
    rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
    atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
    method: A `str`. The algorithm for the black-box ODE solver.
      See documentation for `scipy.integrate.solve_ivp`.
    eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.

  Returns:
    A function that a batch of data points and returns the log-likelihoods in bits/dim,
      the latent code, and the number of function evaluations cost by computation.
  """

  def drift_fn(model, x, t):
    """The drift function of the reverse-time SDE."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=True)
    # Probability flow ODE is a special case of Reverse SDE
    rsde = sde.reverse(score_fn, probability_flow=True)
    return rsde.sde(x, t)[0]

  def div_fn(model, x, t, noise):
    return get_div_fn(lambda xx, tt: drift_fn(model, xx, tt))(x, t, noise)

  def likelihood_fn(model, data, st=None, et=None):
    """Compute an unbiased estimate to the log-likelihood in bits/dim.

    Args:
      model: A score model.
      data: A PyTorch tensor.

    Returns:
      bpd: A PyTorch tensor of shape [batch size]. The log-likelihoods on `data` in bits/dim.
      z: A PyTorch tensor of the same shape as `data`. The latent representation of `data` under the
        probability flow ODE.
      nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
    """
    with torch.no_grad():
      shape = data.shape
      if hutchinson_type == 'Gaussian':
        epsilon = torch.randn_like(data)
      elif hutchinson_type == 'Rademacher':
        epsilon = torch.randint_like(data, low=0, high=2).float() * 2 - 1.
      else:
        raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

      def ode_func(t, x):
        sample = mutils.from_flattened_numpy(x[:-shape[0]], shape).to(data.device).type(torch.float32)
        vec_t = torch.ones(sample.shape[0], device=sample.device) * t
        drift = mutils.to_flattened_numpy(drift_fn(model, sample, vec_t))
        logp_grad = mutils.to_flattened_numpy(div_fn(model, sample, vec_t, epsilon))
        return np.concatenate([drift, logp_grad], axis=0)
      ones = torch.ones((data.shape[0],), device=data.device)
      if st == None or st == 0:
        labels = (ones * 0).type(torch.int64)
        st = eps
        perturbed_data = data
        # labels = (torch.ones((data.shape[0],), device=data.device) * 40).type(torch.int64)
        # sqrt_alphas_cumprod = sde.sqrt_alphas_cumprod.to(data.device)
        # sqrt_1m_alphas_cumprod = sde.sqrt_1m_alphas_cumprod.to(data.device)
        # noise = torch.randn_like(data)
        # perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * data + \
        #                     sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
      elif isinstance(sde, VESDE):
        # st = sde.N - st
        labels = (ones * st).type(torch.int64)
        smld_sigma_array = torch.flip(sde.discrete_sigmas, dims=(0,))
        sigmas = smld_sigma_array.to(data.device)[labels]
        noise = torch.randn_like(data) * sigmas[:, None, None, None]
        perturbed_data = noise + data
        st = st / (sde.N - 1)
      elif isinstance(sde, VPSDE):
        labels = (ones * st).type(torch.int64)
        sqrt_alphas_cumprod = sde.sqrt_alphas_cumprod.to(data.device)
        sqrt_1m_alphas_cumprod = sde.sqrt_1m_alphas_cumprod.to(data.device)
        noise = torch.randn_like(data)
        perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * data + \
                            sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
        st = st / (sde.N - 1)
      if et == None:
        et = sde.T
      else:
        et = et / (sde.N - 1)
      t = labels / (sde.N - 1)
      score_fn = mutils.get_score_fn(sde, model, train=False, continuous=False, use_ebm=False)
      score = score_fn(perturbed_data, t)
      score = score.reshape(score.shape[0], -1).type(torch.float32)

      init = np.concatenate([mutils.to_flattened_numpy(perturbed_data), np.zeros((shape[0],))], axis=0)
      solution = integrate.solve_ivp(ode_func, (st, et), init, rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      zp = solution.y[:, -1]
      z = mutils.from_flattened_numpy(zp[:-shape[0]], shape).to(data.device).type(torch.float32)

    #   labels = ones * et
    #   score = score_fn(z, labels)
    #   return inverse_scaler(z), score
      
      delta_logp = mutils.from_flattened_numpy(zp[-shape[0]:], (shape[0],)).to(data.device).type(torch.float32)
      prior_logp = sde.prior_logp(z)
      bpd = -(prior_logp + delta_logp) / np.log(2)
      N = np.prod(shape[1:])
      bpd = bpd / N
      # A hack to convert log-likelihoods to bits/dim
      offset = 7. - inverse_scaler(-1.)
      bpd = bpd + offset
      return bpd, z, nfe, inverse_scaler(perturbed_data), score

  return likelihood_fn

def forwardProcess(sde, data, st=0, eps=1e-5):
    ones = torch.ones((data.shape[0],), device=data.device)
    if isinstance(sde, VESDE):
        # st = sde.N - st
        labels = (ones * st).type(torch.int64)
        smld_sigma_array = torch.flip(sde.discrete_sigmas, dims=(0,))
        sigmas = smld_sigma_array.to(data.device)[labels]
        noise = torch.randn_like(data) * sigmas[:, None, None, None]
        perturbed_data = noise + data
    elif isinstance(sde, VPSDE):
        labels = (ones * st).type(torch.int64)
        sqrt_alphas_cumprod = sde.sqrt_alphas_cumprod.to(data.device)
        sqrt_1m_alphas_cumprod = sde.sqrt_1m_alphas_cumprod.to(data.device)
        noise = torch.randn_like(data)
        perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * data + \
                            sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
    return perturbed_data


def compute_stats(sde, model, data, inverse_scaler, st=0, eps=1e-5):
    ones = torch.ones((data.shape[0],), device=data.device)
    if isinstance(sde, VESDE):
        # st = sde.N - st
        labels = (ones * st).type(torch.int64)
        smld_sigma_array = torch.flip(sde.discrete_sigmas, dims=(0,))
        sigmas = smld_sigma_array.to(data.device)[labels]
        noise = torch.randn_like(data) * sigmas[:, None, None, None]
        perturbed_data = noise + data
    elif isinstance(sde, VPSDE):
        labels = (ones * st).type(torch.int64)
        sqrt_alphas_cumprod = sde.sqrt_alphas_cumprod.to(data.device)
        sqrt_1m_alphas_cumprod = sde.sqrt_1m_alphas_cumprod.to(data.device)
        noise = torch.randn_like(data)
        perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * data + \
                            sqrt_1m_alphas_cumprod[labels, None, None, None] * noise

    t = labels / (sde.N - 1)
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=False, use_ebm=False)
    score = score_fn(perturbed_data, t)
    score = score.reshape(score.shape[0], -1).type(torch.float32)
    ksd, ksdV = KSD(perturbed_data.reshape(perturbed_data.shape[0], -1), score, score.shape[-1])
    return inverse_scaler(perturbed_data), score, ksd, ksdV

def compute_score(sde, model, data, t=0, noise_idx=0, noise_scale=1.0, use_vp=False, corupt_data=False):
    ones = torch.ones((data.shape[0],), device=data.device)
    if isinstance(sde, VESDE):
        labels = (ones * noise_idx).type(torch.int64)
        smld_sigma_array = torch.flip(sde.discrete_sigmas, dims=(0,))
        sigmas = smld_sigma_array.to(data.device)[labels]
        noise = torch.randn_like(data) * sigmas[:, None, None, None]
        perturbed_data = noise + data
    elif isinstance(sde, VPSDE):
        sqrt_alphas_cumprod = sde.sqrt_alphas_cumprod.to(data.device)
        sqrt_1m_alphas_cumprod = sde.sqrt_1m_alphas_cumprod.to(data.device)
        noise = torch.randn_like(data)
        if use_vp:
            labels = (ones * t).type(torch.int64)
            data = sqrt_alphas_cumprod[labels, None, None, None] * data + \
                   sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
        labels = (ones * noise_idx).type(torch.int64)
        if corupt_data:
            perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * data + \
                             sqrt_1m_alphas_cumprod[labels, None, None, None] * noise * noise_scale
        else:
            perturbed_data = data + sqrt_1m_alphas_cumprod[labels, None, None, None] * noise * noise_scale
        
    t = (ones * t).type(torch.int64) / (sde.N - 1)
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=False, use_ebm=False)
    score = score_fn(perturbed_data, t)
    # labels = (ones * t).type(torch.int64)
    # model_fn = mutils.get_model_fn(model, train=False)
    # score = model_fn(perturbed_data, labels)
    return score

def rbf_kernel(x, dim=2, h=1.):
    # Reference 1: https://github.com/ChunyuanLI/SVGD/blob/master/demo_svgd.ipynb
    # Reference 2: https://github.com/yc14600/svgd/blob/master/svgd.py
    XY = torch.matmul(x, x.T)
    X2_ = torch.reshape(torch.sum(torch.square(x), dim=1), shape=[x.shape[0], 1])
    X2 = X2_.repeat([1, x.shape[0]])
    pdist = torch.sub(torch.add(X2, X2.T), 2 * XY)  # pairwise distance matrix

    kxy = torch.exp(- pdist / h ** 2 / 2.0)  # kernel matrix
    sum_kxy = torch.unsqueeze(torch.sum(kxy, dim=1), 1)
    dxkxy = torch.add(-torch.matmul(kxy, x), torch.multiply(x, sum_kxy)) / (h ** 2)  # sum_y dk(x, y)/dx

    dxykxy_tr = torch.multiply((dim * (h**2) - pdist), kxy) / (h**4)  # tr( dk(x, y)/dxdy )
    return kxy, dxkxy, dxykxy_tr


def imq_kernel(x, dim=2, beta=-.5, c=1.):
    # IMQ kernel
    XY = torch.matmul(x, x.T)
    X2_ = torch.reshape(torch.sum(torch.square(x), dim=1), shape=[x.shape[0], 1])
    X2 = X2_.repeat([1, x.shape[0]])
    pdist = torch.sub(torch.add(X2, X2.T), 2 * XY)  # pairwise distance matrix

    kxy = (c + pdist) ** beta

    coeff = 2 * beta * (c + pdist) ** (beta-1)
    dxkxy = torch.matmul(coeff, x) - torch.multiply(x, torch.unsqueeze(torch.sum(coeff, dim=1), 1))

    dxykxy_tr = torch.multiply((c + pdist) ** (beta - 2),
                            - 2 * dim * c * beta + (- 4 * beta ** 2 + (4 - 2 * dim) * beta) * pdist)

    return kxy, dxkxy, dxykxy_tr

def KSD(x, sq, dim=2, kernel=imq_kernel):
    kxy, dxkxy, dxykxy_tr = kernel(x, dim)
    t13 = torch.multiply(torch.matmul(sq, sq.T), kxy) + dxykxy_tr
    t2 = 2. * torch.trace(torch.matmul(sq, dxkxy.T))
    n = torch.tensor(x.shape[0], dtype=torch.float, device=x.device)

    ksd = (torch.sum(t13) - torch.trace(t13) + t2) / (n * (n-1))
    ksdV = (torch.sum(t13) + t2) / (n ** 2)
    return ksd, ksdV