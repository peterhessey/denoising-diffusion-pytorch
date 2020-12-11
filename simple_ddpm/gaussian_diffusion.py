import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from .helper_functions import exists, default, contains_nan
from tqdm import tqdm


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min = 0, a_max = 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(self, denoise_fn, timesteps=1000, loss_type='l1', betas = None):
        """Gaussian diffusion constructor, the betas and alphas are calculated and stored using buffers 
        during the construction. This as they are not model parameters.

        Args:
            denoise_fn (Unet): The UNet used for predicting noise
            timesteps (int, optional): Number of timesteps. Defaults to 1000.
            loss_type (str, optional): Loss type. Defaults to 'l1'.
            betas ([type], optional): Not sure what this is, perhaps for defining the variances if you want them not-fixed?. Defaults to None.
        """
        super().__init__()
        self.denoise_fn = denoise_fn

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

        # alphas used for loss calculations etc.
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for q(x_t | x_0)
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))

        # calculations for p(x_t | x_{t+1})
        self.register_buffer('recip_sqrt_alphas', to_torch(1. / np.sqrt(alphas)))
        self.register_buffer('noise_standard_deviation', to_torch(np.sqrt(betas)))

    ## Sampling methods

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        """Samples x_t ~ p(x_t| x_{t+1}), using the simple method (Algorithm 2 in Ho et al.)

        Args:
            x (Tensor): Tensor containing batch of x_t+1
            t (Tensor): Tensor containing batch of current time step, len equal to the batch size
            clip_denoised (bool, optional): [description]. Defaults to True.
            repeat_noise (bool, optional): [description]. Defaults to False.

        Returns:
            [Tensor]: Sample of x_t
        """
        b, *_, device = *x.shape, x.device
        sample_extract = partial(extract, t=t, x_shape=x.shape)
 

        recip_sqrt_alpha_t = sample_extract(self.recip_sqrt_alphas)
        beta_t = sample_extract(self.betas)
        sqrt_one_minus_alpha_cumprod_t = sample_extract(self.sqrt_one_minus_alphas_cumprod)

        noise_coef = beta_t / sqrt_one_minus_alpha_cumprod_t

        predicted_noise = self.denoise_fn(x, t)

        if contains_nan(predicted_noise):
            if contains_nan(x):
                None
            print('U-net returning NaN')
            option = input('Display output? (y/n): ').upper()
            if option == 'Y':
                print(predicted_noise)
        # no noise when t == 0
        # nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        langevin_standard_deviation = extract(self.noise_standard_deviation, t, x.shape)
        langevin_noise = langevin_standard_deviation * torch.randn(x.shape, device=device)

        return recip_sqrt_alpha_t * (x - noise_coef * predicted_noise) + langevin_noise


    @torch.no_grad()
    def p_sample_loop(self, shape):
        """Samples p_0 using algorithm 2, as defined in Ho et al. Iteratively calls p_sample, starting with x_T

        Args:
            shape ([Int]): The shape of the image to sample from

        Returns:
            Tensor: p_0 sample
        """
        device = self.betas.device

        # sampling x_T from random noise
        b = shape[0]
        img = torch.randn(shape, device=device)

        # each loop iterates one markov chain step, removing noise slightly each time
        for i in tqdm(reversed(range(1, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img

    @torch.no_grad()
    def sample(self, image_size, batch_size = 16):
        """Calls p_sample_loop

        Args:
            image_size (int): Size of the images to sample
            batch_size (int, optional): Number of images per batch. Defaults to 16.

        Returns:
            Tensor: Sampled batch of images
        """
        return self.p_sample_loop((batch_size, 3, image_size, image_size))
    ## Forward pass methods

    def q_sample(self, x_start, t, noise=None):
        """Sample q(x_t | x_0) for arbitrary t. Algorithm 4 in ho et al.

        Args:
            x_start (Tensor): batch of x_0
            t (Tensor): batch of (randomly generated) timesteps
            noise (tensor, optional): Optional noise tensor. Defaults to None.

        Returns:
            Tensor: batch of x_t ~ q(x_t|x_0)
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )


    def p_losses(self, x_start, t, noise = None):
        """Calculates the loss for the given batch

        Args:
            x_start (Tensor): Batch of training images
            t (Tensor): Randomly selected timesteps tensor, len = batch size
            noise (Tensor, optional): Optional input noise. Defaults to None.

        Raises:
            NotImplementedError: Raised if invalid loss method selected

        Returns:
            Float: Loss value
        """
        
        noise = default(noise, lambda: torch.randn_like(x_start))

        # sample x_t from q(x_t|x_0)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t)

        if self.loss_type == 'l1':
            loss = (noise - x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss


    def forward(self, x, *args, **kwargs):
        """Forward pass function. Selects time steps uniformly, calculates and returns UNet loss.

        Args:
            x (Tensor): Tensor containing image batch

        Returns:
            Point: Loss value
        """
        
        b, *_, device = *x.shape, x.device
        # selects random timesteps to sample from
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, *args, **kwargs)