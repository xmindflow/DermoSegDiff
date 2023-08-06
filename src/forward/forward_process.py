import torch


class ForwardProcess(object):
    def __init__(self, forward_schedule):
        self.forward_schedule = forward_schedule

    # forward diffusion (using the nice property)
    def q_sample(self, x_start, t, noise=None):
        # if t > self.timesteps:
        #     # raise ValueError("'t' must be equal or less than 'T'.")
        #     t=self.timesteps

        if noise is None:
            noise = torch.randn_like(x_start)

        ex = self.forward_schedule.extract(t, x_start.shape)
        sqrt_alphas_cumprod_t = ex["sqrt_alphas_cumprod"]
        sqrt_one_minus_alphas_cumprod_t = ex["sqrt_one_minus_alphas_cumprod"]

        noisy_x = (
            sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        )

        return noisy_x
