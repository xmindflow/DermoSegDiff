import torch
from tqdm import tqdm
from models import *



@torch.no_grad()
def p_sample(forward_schedule, model, images, x, t, t_index):
    fs = forward_schedule

    exs = fs.extract(t, x.shape)
    betas_t = exs["betas"]
    sqrt_one_minus_alphas_cumprod_t = exs["sqrt_one_minus_alphas_cumprod"]
    sqrt_recip_alphas_t = exs["sqrt_recip_alphas"]

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean

    if isinstance(model, DermoSegDiff):
        predicted_noise = model(x, images, t)
    elif isinstance(model, Baseline):
        predicted_noise = model(x, t, images)
    else:
        NotImplementedError('given model is unknown!')

    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = exs["posterior_variance"]
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise


# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(forward_schedule, model, images, out_channels, desc=""):
    timesteps = forward_schedule.timesteps
    device = next(model.parameters()).device

    shape = (images.shape[0], out_channels, images.shape[2], images.shape[3])
    b = shape[0]

    # start from pure noise (for each example in the batch)
    msk = torch.randn(shape, device=device)
    msks = []

    desc = f"{desc} - sampling loop time step" if desc else "sampling loop time step"
    for i in tqdm(
        reversed(range(0, timesteps)), desc=desc, total=timesteps, leave=False
    ):
        msk = p_sample(
            forward_schedule,
            model,
            images,
            msk,
            torch.full((b,), i, device=device, dtype=torch.long),
            i,
        )
        msks.append(msk.detach().cpu())
    return msks


@torch.no_grad()
def sample(forward_schedule, model, images, out_channels=2, desc=None):
    return p_sample_loop(forward_schedule, model, images, out_channels, desc)



def reverse_by_epsilon(forward_process, predicted_noise, x, t):
    fs = forward_process.forward_schedule
    exs = fs.extract(t, x.shape)
    
    betas_t = exs["betas"]
    sqrt_one_minus_alphas_cumprod_t = exs["sqrt_one_minus_alphas_cumprod"]
    sqrt_recip_alphas_t = exs["sqrt_recip_alphas"]
    posterior_variance_t = exs["posterior_variance"]

    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
    )
    
    noise = torch.randn_like(x) if t[0].item() > 0 else 0
    # Algorithm 2 line 4:
    res = model_mean + torch.sqrt(posterior_variance_t) * noise

    return res

