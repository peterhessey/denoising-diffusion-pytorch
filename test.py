#from simple_ddpm import Unet, GaussianDiffusion, Trainer
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).cuda()

diffusion = GaussianDiffusion(
    model,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    './data/cifar10/cifar10/train/cat',
    image_size = 32,
    train_batch_size = 32,
    train_lr = 1e-4,
    train_num_steps = 700000,         # total training steps
    save_and_sample_every= 100,      # how often to sample
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False                       # turn on mixed precision training with apex
)

trainer.train()

# test my own sample method

# trainer.load('699')
# trainer.sample_images()