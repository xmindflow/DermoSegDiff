import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from modules.schedules import (
    linear_beta_schedule,
    quadratic_beta_schedule,
    cosine_beta_schedule,
    sigmoid_beta_schedule,
)


class ForwardSchedule(object):
    def __init__(self, timesteps, beta_start=0.0001, beta_end=0.02, mode="linear"):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.mode = mode.lower()
        # start calculation
        self.calc_vars()

    def get_scheduler(self):
        if self.mode == "linear":
            return linear_beta_schedule
        elif self.mode == "quadratic":
            return quadratic_beta_schedule
        elif self.mode == "cosine":
            return cosine_beta_schedule
        elif self.mode == "sigmoid":
            return sigmoid_beta_schedule
        else:
            raise ValueError(
                "Schedule mode must be in: [linear,quadratic,cosine,sigmoid]"
            )

    def calc_vars(
        self,
    ):
        # define beta schedule
        scheduler = self.get_scheduler()
        self.betas = scheduler(
            timesteps=self.timesteps, beta_start=self.beta_start, beta_end=self.beta_end
        )

        # define alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def __extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def extract(self, t, x_shape):
        return {
            "betas": self.__extract(self.betas, t, x_shape),
            "sqrt_alphas_cumprod": self.__extract(self.sqrt_alphas_cumprod, t, x_shape),
            "sqrt_one_minus_alphas_cumprod": self.__extract(
                self.sqrt_one_minus_alphas_cumprod, t, x_shape
            ),
            "sqrt_recip_alphas": self.__extract(self.sqrt_recip_alphas, t, x_shape),
            "posterior_variance": self.__extract(self.posterior_variance, t, x_shape),
        }
