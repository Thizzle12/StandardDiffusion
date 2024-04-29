import logging
from typing import List, Tuple

import torch
from torch import nn
from tqdm import tqdm


class Diffusion:

    def __init__(
        self,
        T: int = 500,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        img_size: Tuple[int, int] = (32, 32),
        channels: int = 3,
        device: str = "cpu",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.channels = channels

        self.betas = torch.linspace(beta_start, beta_end, T).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(
        self,
        x: torch.Tensor,
        t: int,
    ) -> torch.Tensor:

        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])
        sqrt_alpha_bar = sqrt_alpha_bar[:, None, None, None]

        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alphas[t])
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar[:, None, None, None]

        noise = torch.normal(0.0, 1.0, x.shape).to(self.device)
        assert noise.shape == x.shape, f"Noise shape {noise.shape} != input image shape {x.shape}"

        x_noise = x * sqrt_alpha_bar + noise * sqrt_one_minus_alpha_bar

        return x_noise, noise

    def p_mean_std(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: int,
    ) -> torch.Tensor:

        beta = self.betas[t][:, None, None, None]
        alpha = self.alphas[t][:, None, None, None]
        alpha_bar = self.alpha_bar[t][:, None, None, None]

        predicted_noise = model(x_t, t)
        mean = (1.0 / torch.sqrt(alpha)) * x_t - (beta / torch.sqrt(1 - alpha_bar)) * predicted_noise
        std = torch.sqrt(beta)

        return mean, std

    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: int,
    ) -> torch.Tensor:

        mean, std = self.p_mean_std(model, x_t, t)
        noise = torch.normal(0.0, 1.0, x_t.shape).to(self.device)
        x_t_prev = mean + std * noise

        return x_t_prev

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample timesteps from the diffusion process uniformly for training.

        Args:
            batch_size (int): _description_

        Returns:
            torch.Tensor: _description_
        """

        return torch.randint(low=1, high=self.T, size=(batch_size,)).to(self.device)

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        batch_size: int,
        timesteps_to_save: List,
    ) -> torch.Tensor:

        logging.info(f"Sampling {batch_size} images from the diffusion process")
        model.eval()

        if timesteps_to_save is not None:
            intermediate_samples = []
        x = torch.randn(batch_size, self.channels, self.img_size, self.img_size).to(self.device)
        with tqdm(reversed(range(1, self.T)), position=0, total=self.T - 1) as pbar:
            for i, _ in enumerate(pbar):
                pbar.set_description(f"Sampling from timestep {i + 1}/{self.T}")
                t = (torch.ones(batch_size) * i).long().to(self.device)
                x = self.p_sample(model, x, t)

                if timesteps_to_save is not None and i in timesteps_to_save:
                    x_intermediate = (x.clamp(-1, 1) + 1) / 2
                    x_intermediate = (x_intermediate * 255).clamp(0, 255).to(torch.uint8)
                    intermediate_samples.append(x_intermediate)

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).clamp(0, 255).to(torch.uint8)

        if timesteps_to_save is not None:
            return x, intermediate_samples
        else:
            return x
