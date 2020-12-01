from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).cuda()

diffusion = GaussianDiffusion(
    model,
    timesteps = 1000,   # number of step!s
    loss_type = 'l1'    # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    './data/cifar10/cifar10/train/cat',
    image_size = 32,
    train_batch_size = 32,
    train_lr = 2e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = True                       # turn on mixed precision training with apex
)

trainer.train()