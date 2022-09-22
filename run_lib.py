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
"""Training and evaluation for score-based generative models. """

import os

import numpy as np
import logging
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp, wrn, wideresnet
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import likelihood
import sde_lib
from absl import flags
import torch
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint
from einops import rearrange
from patchify import patchify, unpatchify

FLAGS = flags.FLAGS


def train(config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  os.makedirs(sample_dir, exist_ok=True) 

  # Initialize model.
  score_model = mutils.create_model(config)
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  os.makedirs(checkpoint_dir, exist_ok=True) 
  os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True) 
  # Resume training when intermediate checkpoints are detected
  state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
  initial_step = int(state['step'])

  # Build data iterators
  train_ds, eval_ds, _ = datasets.get_dataset(config,
                                              uniform_dequantization=config.data.uniform_dequantization)
  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  use_ebm = config.ebm.use_ebm
  train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting, use_ebm=use_ebm)
  eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting, use_ebm=use_ebm)

  # Building sampling functions
  if config.training.snapshot_sampling:
    sampling_shape = (config.training.batch_size, config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps, use_ebm)

  num_train_steps = config.training.n_iters

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  logging.info("Starting training loop at step %d." % (initial_step,))

  for step in range(initial_step, num_train_steps + 1):
    # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
    batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device).float()
    batch = batch.permute(0, 3, 1, 2)
    batch = scaler(batch)
    # Execute one training step
    loss = train_step_fn(state, batch)
    if step % config.training.log_freq == 0:
      logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))

    # Save a temporary checkpoint to resume training after pre-emption periodically
    if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
      save_checkpoint(checkpoint_meta_dir, state)

    # Report the loss on an evaluation dataset periodically
    if step % config.training.eval_freq == 0:
      eval_batch = torch.from_numpy(next(eval_iter)['image']._numpy()).to(config.device).float()
      eval_batch = eval_batch.permute(0, 3, 1, 2)
      eval_batch = scaler(eval_batch)
      eval_loss = eval_step_fn(state, eval_batch)
      logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))

    # Save a checkpoint periodically and generate samples if needed
    if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
      # Save the checkpoint.
      save_step = step // config.training.snapshot_freq
      save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

      # Generate and save samples
      if config.training.snapshot_sampling:
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())
        sample, n = sampling_fn(score_model)
        ema.restore(score_model.parameters())
        this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
        os.makedirs(this_sample_dir, exist_ok=True)
        nrow = int(np.sqrt(sample.shape[0]))
        image_grid = make_grid(sample, nrow, padding=2)
        sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
        np.save(os.path.join(this_sample_dir, "sample.np"), sample)
        save_image(image_grid, os.path.join(this_sample_dir, "sample.png"))


def evaluate(config,
             workdir,
             eval_folder="eval"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  os.makedirs(eval_dir, exist_ok=True)

  # Build data pipeline
  train_ds, eval_ds, _ = datasets.get_dataset(config,
                                              uniform_dequantization=config.data.uniform_dequantization,
                                              evaluation=True)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  score_model = mutils.create_model(config)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  checkpoint_dir = os.path.join(workdir, "checkpoints")

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Create the one-step evaluation function when loss computation is enabled
  if config.eval.enable_loss:
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    likelihood_weighting = config.training.likelihood_weighting
    use_ebm = config.ebm.use_ebm

    reduce_mean = config.training.reduce_mean
    eval_step = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                   reduce_mean=reduce_mean,
                                   continuous=continuous,
                                   likelihood_weighting=likelihood_weighting,
                                   use_ebm=use_ebm)


  # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
  train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config,
                                                      uniform_dequantization=True, evaluation=True)
  if config.eval.bpd_dataset.lower() == 'train':
    ds_bpd = train_ds_bpd
    bpd_num_repeats = 1
  elif config.eval.bpd_dataset.lower() == 'test':
    # Go over the dataset 5 times when computing likelihood on the test dataset
    ds_bpd = eval_ds_bpd
    bpd_num_repeats = 5
  else:
    raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")

  # Build the likelihood computation function when likelihood is enabled
  if config.eval.enable_bpd:
    likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler)

  # Build the sampling function when sampling is enabled
  if config.eval.enable_sampling:
    use_ebm = config.ebm.use_ebm
    sampling_shape = (config.eval.batch_size,
                      config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps, use_ebm)

  # Use inceptionV3 for images with resolution higher than 256.
#   inceptionv3 = config.data.image_size >= 256
#   inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

  begin_ckpt = config.eval.begin_ckpt
  logging.info("begin checkpoint: %d" % (begin_ckpt,))
  for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
    # Wait if the target checkpoint doesn't exist yet
    waiting_message_printed = False
    ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
    if not os.path.exists(ckpt_filename):
        logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
        return

    # Wait for 2 additional mins in case the file exists but is not ready for reading
    ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
    state = restore_checkpoint(ckpt_path, state, device=config.device)
    ema.copy_to(score_model.parameters())
    # Compute the loss function on the full evaluation dataset if loss computation is enabled
    if config.eval.enable_loss:
      all_losses = []
      eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
      for i, batch in enumerate(eval_iter):
        eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
        eval_batch = eval_batch.permute(0, 3, 1, 2)
        eval_batch = scaler(eval_batch)
        eval_loss = eval_step(state, eval_batch)
        all_losses.append(eval_loss.item())
        if (i + 1) % 1000 == 0:
          logging.info("Finished %dth step loss evaluation" % (i + 1))

      # Save loss values to disk or Google Cloud Storage
      all_losses = np.asarray(all_losses)
      np.savez_compressed(os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz"), all_losses=all_losses, mean_loss=all_losses.mean())


    # Compute log-likelihoods (bits/dim) if enabled
    if config.eval.enable_bpd:
      bpds = []
      for repeat in range(bpd_num_repeats):
        bpd_iter = iter(ds_bpd)  # pytype: disable=wrong-arg-types
        for batch_id in range(len(ds_bpd)):
          batch = next(bpd_iter)
          eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
          eval_batch = eval_batch.permute(0, 3, 1, 2)
          eval_batch = scaler(eval_batch)
          bpd = likelihood_fn(score_model, eval_batch)[0]
          bpd = bpd.detach().cpu().numpy().reshape(-1)
          bpds.extend(bpd)
          logging.info(
            "ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f" % (ckpt, repeat, batch_id, np.mean(np.asarray(bpds))))
          bpd_round_id = batch_id + len(ds_bpd) * repeat
          # Save bits/dim to disk or Google Cloud Storage
          np.savez_compressed(os.path.join(eval_dir,
                                f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz"), bpd)

    # Generate samples and compute IS/FID/KID when enabled
    if config.eval.enable_sampling:
      num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
      for r in range(num_sampling_rounds):
        logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))

        # Directory to save samples. Different for each host to avoid writing conflicts
        this_sample_dir = os.path.join(
          eval_dir, f"ckpt_{ckpt}")
        os.makedirs(this_sample_dir, exist_ok=True)
        samples, n = sampling_fn(score_model)
        nrow = int(np.sqrt(samples.shape[0]))
        image_grid = make_grid(samples, nrow, padding=2)
        samples = np.clip(samples.permute(0, 2, 3, 1).detach().cpu().numpy() * 255., 0, 255).astype(np.uint8)
        samples = samples.reshape(
          (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
        # Write samples to disk or Google Cloud Storage
        np.save(os.path.join(this_sample_dir, f"samples_{r}.np"), samples)
        save_image(image_grid, os.path.join(this_sample_dir, f"samples_{r}.png"))
        
  return

def evaluate_ood(config,
             workdir,
             eval_folder="eval_ood"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  os.makedirs(eval_dir, exist_ok=True)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  score_model = mutils.create_model(config)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  checkpoint_dir = os.path.join(workdir, "checkpoints")

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")


  # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
  train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config,
                                                      uniform_dequantization=True, evaluation=True)
  if config.eval.bpd_dataset.lower() == 'train':
    ds_bpd = train_ds_bpd
    bpd_num_repeats = 1
  elif config.eval.bpd_dataset.lower() == 'test':
    # Go over the dataset 5 times when computing likelihood on the test dataset
    ds_bpd = eval_ds_bpd
    bpd_num_repeats = 5
  else:
    raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")

  # Build the likelihood computation function when likelihood is enabled
  likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler)
  sampling_shape = (config.eval.batch_size,
                    config.data.num_channels,
                    config.data.image_size, config.data.image_size)
  sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  begin_ckpt = config.eval.begin_ckpt
  logging.info("begin checkpoint: %d" % (begin_ckpt,))
  for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
    # Wait if the target checkpoint doesn't exist yet
    ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
    if not os.path.exists(ckpt_filename):
        logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
        return

    ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
    state = restore_checkpoint(ckpt_path, state, device=config.device)
    ema.copy_to(score_model.parameters())

    bpd_iter = iter(ds_bpd)  # pytype: disable=wrong-arg-types
    # for batch_id in range(len(ds_bpd)):
    #     norms = []
    #     scores = []
    #     batch = next(bpd_iter)
    #     eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
    #     eval_batch = eval_batch.permute(0, 3, 1, 2)
    #     eval_batch = scaler(eval_batch)
    #     for t in range(0, 1000, 1000):
    #         for idx in range(0, 1000, 2):
    #             score = likelihood.compute_score(sde, score_model, eval_batch, idx, t)
    #             scores.append(score.reshape(score.shape[0], -1).cpu().detach().numpy())
    #             norms.append(torch.norm(score.reshape(score.shape[0], -1), dim=-1).mean().cpu().detach().numpy())
    #             logging.info(
    #                 "batch: %d, ckpt: %d, score t: %d, image t: %d, score norm: %3f" % (batch_id, ckpt, t, idx, torch.norm(score.reshape(score.shape[0], -1), dim=-1).mean()))
    #         np.save(os.path.join(eval_dir, f"{config.eval.bpd_dataset}_ckpt_{ckpt}_batch_{batch_id}_st_{t}_s2.np"), np.asarray(norms))
    #         np.save(os.path.join(eval_dir, f"{config.eval.bpd_dataset}_ckpt_{ckpt}_batch_{batch_id}_st_{t}_s.np"), np.asarray(scores))

    batch = next(bpd_iter)
    batch = batch['image']._numpy()
    batch = transform(batch)
    eval_batch = torch.from_numpy(batch).to(config.device).float()
    eval_batch = eval_batch.permute(0, 3, 1, 2)
    eval_batch = scaler(eval_batch)

    # scores = []
    # for t in range(0, 1000, 20):
    #     x_t, score_t = likelihood_fn(score_model, eval_batch, st=t, num_scales=config.model.num_scales)
    #     scores.append(score_t.reshape(score_t.shape[0], -1).cpu().detach().numpy())
    #     recon_samples, n = sampling_fn(score_model, t, scaler(x_t))
    #     nrow = int(np.sqrt(x_t.shape[0]))
    #     image_grid = make_grid(x_t, nrow, padding=2)
    #     save_image(image_grid, os.path.join(eval_dir, f"sample_shuffle_time_{t}.png"))
    #     image_grid = make_grid(recon_samples, nrow, padding=2)
    #     save_image(image_grid, os.path.join(eval_dir, f"sample_shuffle_recon_time_{t}.png"))
    #     logging.info('Timestep: %d' % (t))
    # np.save(os.path.join(eval_dir, f"{config.eval.bpd_dataset}_ckpt_{ckpt}_shuffle_score.np"), np.asarray(scores))

    images = []
    scores = []
    multi_ksd = []
    multi_ksdV = []
    for t in range(0, 1000, 10):
        perturbed_data, score, ksd, ksdV = likelihood_fn(score_model, eval_batch, st=t, num_scales=config.model.num_scales)
        samples = np.clip(perturbed_data.permute(0, 2, 3, 1).detach().cpu().numpy() * 255., 0, 255).astype(np.uint8)
        samples = samples.reshape((-1, config.data.image_size, config.data.image_size, config.data.num_channels))
        images.append(samples)
        scores.append(score.reshape(score.shape[0], -1).cpu().detach().numpy())
        multi_ksd.append(ksd.detach().cpu().numpy())
        multi_ksdV.append(ksdV.detach().cpu().numpy())
        logging.info('Timestep: %d' % (t))
    res = {'image': np.asarray(images), 'score': np.asarray(scores), 'ksd': np.asarray(multi_ksd), 'ksdV': np.asarray(multi_ksdV)}
    np.save(os.path.join(eval_dir, f"{config.eval.bpd_dataset}_ckpt_{ckpt}_stats.npy"), res)


    # scores = []
    # for t in range(0, 1000, 100):
    #     for idx in range(10):
    #         score = likelihood.compute_score(sde, score_model, eval_batch, t, t)
    #         scores.append(score.reshape(score.shape[0], -1).cpu().detach().numpy())
    #     logging.info('Timestep: %d' % (t))
    #     np.save(os.path.join(eval_dir, f"{config.eval.bpd_dataset}_ckpt_{ckpt}_st_{t}_single_score.np"), np.asarray(scores))

    # scores = []
    # for t in range(1, 1000, 100):
    #     x_t, score_t = likelihood_fn(score_model, eval_batch, st=0, et=t, num_scales=config.model.num_scales)
    #     scores.append(score_t.reshape(score_t.shape[0], -1).cpu().detach().numpy())
    #     nrow = int(np.sqrt(x_t.shape[0]))
    #     image_grid = make_grid(x_t, nrow, padding=2)
    #     save_image(image_grid, os.path.join(eval_dir, f"sample_ode_time_{t}.png"))
    #     logging.info('Timestep: %d' % (t))
    # np.save(os.path.join(eval_dir, f"{config.eval.bpd_dataset}_ckpt_{ckpt}_ode_score.np"), np.asarray(scores))


    # st, et, step = 0, 1000, 200
    # # Compute log-likelihoods (bits/dim) if enabled
    # for repeat in range(bpd_num_repeats):
    #     bpd_iter = iter(ds_bpd)  # pytype: disable=wrong-arg-types
    #     for batch_id in range(len(ds_bpd)):
    #       bpds = []
    #       batch = next(bpd_iter)
    #       eval_batch = torch.from_numpy(batch).to(config.device).float()
    #       eval_batch = eval_batch.permute(0, 3, 1, 2)
    #       eval_batch = scaler(eval_batch)
    #       for idx in range(st, et, step):
    #         bpd, latent_z, _, perturbed_data, score = likelihood_fn(score_model, eval_batch, st=idx, num_scales=config.model.num_scales)
    #         # recon_samples, n = sampling_fn(score_model, idx, scaler(perturbed_data))
    #         # recon_samples, n = sampling_fn(score_model, latent_z)
    #         bpd = bpd.detach().cpu().numpy().reshape(-1)
    #         bpds.append(bpd.mean())
    #         logging.info(
    #             "ckpt: %d, repeat: %d, batch: %d, timestep: %d, mean bpd: %6f, score norm: %3f" % (ckpt, repeat, batch_id, idx, np.mean(np.asarray(bpd)), torch.norm(score.reshape(score.shape[0], -1), dim=-1).mean()))


    #         # bpd, _, _, _ = likelihood_fn(score_model, scaler(recon_samples), 0, num_scales=config.model.num_scales)
    #         # bpd = bpd.detach().cpu().numpy().reshape(-1)
    #         # logging.info(
    #             # "ckpt: %d, repeat: %d, batch: %d, timestep: %d, recon mean bpd: %6f" % (ckpt, repeat, batch_id, idx, np.mean(np.asarray(bpd))))
    #         bpd_round_id = batch_id + len(ds_bpd) * repeat
    #         if batch_id == 0:
    #             nrow = int(np.sqrt(perturbed_data.shape[0]))
    #             image_grid = make_grid(perturbed_data, nrow, padding=2)
    #             samples = np.clip(perturbed_data.permute(0, 2, 3, 1).detach().cpu().numpy() * 255., 0, 255).astype(np.uint8)
    #             samples = samples.reshape(
    #                 (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
    #             save_image(image_grid, os.path.join(eval_dir, f"sample_time_{idx}.png"))

    #             # image_grid = make_grid(recon_samples, nrow, padding=2)
    #             # samples = np.clip(recon_samples.permute(0, 2, 3, 1).detach().cpu().numpy() * 255., 0, 255).astype(np.uint8)
    #             # samples = samples.reshape(
    #                 # (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
    #             # save_image(image_grid, os.path.join(eval_dir, f"recon_sample_time_{idx}.png"))

    #       # Save bits/dim to disk or Google Cloud Storage
    #       np.save(os.path.join(eval_dir, f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}_{st}_{et}_{step}.np"), np.asarray(bpds))

def transform(images):
    for i in range(images.shape[0]):
        image = rearrange(patchify(images[i], (16, 16, 3), step=16), 'n1 n2 n3 p1 p2 c -> (n1 n2 n3) p1 p2 c')
        per = np.random.permutation(image.shape[0])
        # print(per)
        image = image[per, :, :, :]
        image = rearrange(image, '(n1 n2 n3) p1 p2 c -> n1 n2 n3 p1 p2 c', n1=2, n2=2)
        images[i] = unpatchify(image, (32, 32, 3))
    return images