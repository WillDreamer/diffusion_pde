# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for Latte using PyTorch DDP.
"""

import torch
# Maybe use fp16 percision training need to set to False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import io
import os
import math
import argparse

import torch.distributed as dist
from glob import glob
from time import time
from copy import deepcopy
from einops import rearrange
from models import get_models
from datasets.CNS_data_utils import DatasetSingle
from diffusion import create_diffusion
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers.optimization import get_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils import (clip_grad_norm_, create_logger, update_ema, 
                   requires_grad, cleanup, create_tensorboard, 
                   write_tensorboard, setup_distributed,)
import numpy as np
from transformers import T5EncoderModel, T5Tokenizer

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    setup_distributed()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    seed = args.global_seed + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, local rank={local_rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        save_dir = args.save_dir
        experiment_dir = f"{save_dir}/train/{args.exp_name}"  # Stores saved model checkpoints
        ckpt_dir = f"{save_dir}/train/{args.ckpt_name}"
        # tb_dir = f"{save_dir}/log_tb/{args.exp_name}"  # Stores tensorboard logs
        tb_dir = f"/home/whx/diffusion_pde/log/{args.exp_name}"  # The original tensorboard log path doesn't work for vscode now, idk why
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(tb_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        tb_writer = create_tensorboard(tb_dir)
        OmegaConf.save(args, os.path.join(experiment_dir, 'config.yaml'))
        os.system(f"cp -r /home/whx/diffusion_pde/log/foundation_pde {experiment_dir}")
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)
        tb_writer = None

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    # I cringed.
    # assert args.use_coordinates, "Who uses positional encodings in PDE?" 
    sample_size = args.image_size
    args.latent_size = sample_size
    args.num_classes = len(args.filename)
    if args.direct_predict:
        args.pred_xstart = True
        args.learn_sigma = False
    model = get_models(args)
    # Note that parameter initialization is done within the Latte constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="", predict_xstart=args.pred_xstart, direct_predict=args.direct_predict,
                                 learn_sigma=args.learn_sigma)  # default: 1000 steps, linear noise schedule


    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        # TODO, need to checkout
        # Get the most recent checkpoint
        dirs = os.listdir(ckpt_dir)
        dirs = [d for d in dirs if d.endswith("pt")]
        dirs = sorted(dirs, key=lambda x: int(x.split(".")[0]))
        path = f"{ckpt_dir}/{dirs[-1]}"
        logger.info(f"Resuming from checkpoint {path}")
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        train_steps = int(path.split(".")[0].split("/")[-1])

    if args.use_compile:
        model = torch.compile(model)

    # set distributed training
    model = DDP(model.to(device), device_ids=[local_rank])

    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    if args.resume_from_checkpoint:
        opt.load_state_dict(checkpoint["opt"])

    # Freeze vae and text_encoder
    # vae.requires_grad_(False)

    # Setup data:
    # tr_data = PDEBench_npy(
    #     filename=args.filename,
    #     num_frames=args.num_initial_steps+args.num_future_steps,
    #     # image_size=args.image_size,
    #     # num_channels=args.in_channels,
    #     normalize=args.normalize,
    #     use_spatial_sample=args.use_spatial_sample,
    #     use_coordinates=args.use_coordinates,
    #     # frame_interval=args.frame_interval,
    #     # data_path=args.root,
    #     is_train=True,
    #     # train_ratio=args.train_ratio,
    # )
    # val_data = PDEBench(
    #     # filename=args.flnm,
    #     # num_frames=args.initial_steps+args.future_steps,
    #     # image_size=args.image_size,
    #     # num_channels=args.in_channels,
    #     normalize=args.normalize,
    #     # use_coordinates=args.use_coordinates,
    #     # frame_interval=args.frame_interval,
    #     # root=args.root,
    #     is_train=False,
    #     # train_ratio=args.train_ratio,
    # )
    data_args = {
        "PDE_type": "CNS",  # choices: ['CNS', 'DR']
        "data_path": "/local2/shared_data/DBench_data/PDEBench/",
        "data_set": "2D_CFD_Turb_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5",
        "input_size": 128,
        "in_chans": 4,
        "reduced_resolution": 4,
        "reduced_resolution_t": 1,
        "reduced_batch": 1,
        "initial_step": 4
        }
    
    from argparse import Namespace

    data_args = Namespace(**data_args)
    tr_data = DatasetSingle(args = data_args)

    sampler = DistributedSampler(
        tr_data,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
        )
    tr_loader = DataLoader(
        tr_data,
        batch_size=int(args.local_batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    # val_loader = DataLoader(
    #     val_data,
    #     batch_size=int(args.local_batch_size),
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     drop_last=True
    # )
    logger.info(f"Dataset contains {len(tr_data):,} videos ({args.data_path})")

    # Scheduler
    lr_scheduler = get_scheduler(
        name="constant",
        optimizer=opt,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    if not args.resume_from_checkpoint:
        train_steps = 0 
    log_steps = 0
    running_loss = 0
    first_epoch = 0
    start_time = time()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(tr_loader))
    first_epoch = train_steps // num_update_steps_per_epoch
    resume_step = train_steps % num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if args.pretrained:
        train_steps = int(args.pretrained.split("/")[-1].split('.')[0])

    condition_by_cleaninitial = None
    if args.num_initial_steps > 0 and "Latte" in args.model:
        condition_by_cleaninitial = args.num_initial_steps
        print('condition_by_cleaninitial:', condition_by_cleaninitial)

    for epoch in range(first_epoch, num_train_epochs):
        sampler.set_epoch(epoch)
        for step, video_data in enumerate(tr_loader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                continue

            initial_step = video_data[0].to(device, non_blocking=True) # [b h w f c]->[b f c h w]
            x = video_data[1].to(device, non_blocking=True) # [b f c h w]

            initial_step = initial_step.permute(0, 3, 4, 1, 2).to(device, non_blocking=True)
            x = x.permute(0, 3, 4, 1, 2).to(device, non_blocking=True)
            grid = video_data[-1].to(device, non_blocking=True) if args.use_coordinates else None # [b 2 h w]

            model_kwargs = dict(y=None)
            model_kwargs["initial_steps"] = initial_step
            model_kwargs["grid"] = grid

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            loss_dict = diffusion.training_losses(model, x, t, condition_by_cleaninitial, model_kwargs)
            loss = loss_dict["loss"].mean()
            loss.backward()

            if train_steps < args.start_clip_iter: # if train_steps >= start_clip_iter, will clip gradient
                gradient_norm = clip_grad_norm_(model.module.parameters(), args.clip_max_norm, clip_grad=False)
            else:
                gradient_norm = clip_grad_norm_(model.module.parameters(), args.clip_max_norm, clip_grad=True)

            opt.step()
            lr_scheduler.step()
            opt.zero_grad()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                # logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                logger.info(f"(step={train_steps:07d}/epoch={epoch:04d}) Train Loss: {avg_loss:.4f}, Gradient Norm: {gradient_norm:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                write_tensorboard(tb_writer, 'Train Loss', avg_loss, train_steps)
                write_tensorboard(tb_writer, 'Gradient Norm', gradient_norm, train_steps)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save Latte checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{experiment_dir}/{train_steps:07d}.pt"
                    # checkpoint_path = f"{experiment_dir}/model_last.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train Latte with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/home/whx/diffusion_pde/configs/pdebench/pdebench_train.yaml")
    args = parser.parse_args()
    main(OmegaConf.load(args.config))
