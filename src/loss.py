import torch
import torch.nn.functional as F
from utils.helper_funcs import calc_boundary_att
from common.logging import get_logger
from models import *



calc_boundary = None


class BoundaryLoss(torch.nn.Module):
    """Some Information about BoundaryLoss"""
    def __init__(self, parameters={}):
        super().__init__()
        self.logger = get_logger()
        
        self.gamma=parameters.get("gamma", 1.5)
        root=parameters.get("root", "l2")
        if root == "l2":
            self.calc_root_loss = lambda p, t: torch.abs(p-t)**2
        elif root == "l1":
            self.calc_root_loss = lambda p, t: torch.abs(p-t)
        else:
            self.logger.exception("Not implemented!")

    def forward(self, x, t, T, predicted_noise, noise):
        boundary_att = calc_boundary_att(x, t, T=T, gamma=self.gamma)
        root_loss = self.calc_root_loss(predicted_noise, noise)
        return (boundary_att * root_loss).mean()


def p_losses(
    forward_process,
    denoise_model,
    x_start, # target
    g, # guidance
    t,
    cfg,
    noise=None,
):
    global calc_boundary
    logger = get_logger()
    
    T = cfg["diffusion"]["schedule"]["timesteps"]

    cfg_loss = cfg['training']['loss']
    loss_type = cfg['training']['loss_name']

    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = forward_process.q_sample(x_start=x_start, t=t, noise=noise)
    
    if x_noisy.isnan().any().item() or g.isnan().any().item():
        print(f"\nt:{t.detach().cpu().numpy()}, x_start:{x_start.isnan().any().item()}, x_noisy:{x_noisy.isnan().any().item()}, g:{g.isnan().any().item()}\n")
    
    if isinstance(denoise_model, DermoSegDiff):
        predicted_noise = denoise_model(x_noisy, g, t)
    elif isinstance(denoise_model, Baseline):
        predicted_noise = denoise_model(x=x_noisy, time=t, x_self_cond=g)
    else:
        logger.exception('given <denoise_model> is unknown!')
    
    if x_noisy.isnan().any().item() or g.isnan().any().item() or predicted_noise.isnan().any().item():
        print(f"\n\nx: {x_noisy.isnan().any().item()}, g: {g.isnan().any().item()}, preds: {predicted_noise.isnan().any().item()}\n\n")
        
    losses=dict()
    ln = loss_type.lower()
    if ln=="l1" or (ln=="hybrid" and "l1" in cfg_loss.keys()):
        losses["l1"] = F.l1_loss(predicted_noise, noise)
    if ln=="l2" or (ln=="hybrid" and "l2" in cfg_loss.keys()):
        losses["l2"] = F.mse_loss(predicted_noise, noise)
    if ln=="huber" or (ln=="hybrid" and "huber" in cfg_loss.keys()):
        losses["huber"] = F.smooth_l1_loss(predicted_noise, noise)
    if ln=="boundary" or (ln=="hybrid" and "boundary" in cfg_loss.keys()):
        if not calc_boundary:
            parameters = cfg_loss["boundary"].get("params", {})
            calc_boundary = BoundaryLoss(parameters)
        losses["boundary"] = calc_boundary(x_start, t, T, predicted_noise, noise)
  
    if loss_type.lower() in ["l1", "l2", "huber", "boundary"]:
        loss = losses[loss_type.lower()]
    else: # loss_type == "hybrid":
        for k in cfg_loss.keys():
            if k in losses.keys(): continue
            logger.exception("Not implemented loss!")
        loss = 0
        for l_name, l_d in cfg_loss.items():
            loss += l_d.get("cofficient", 1)*losses[l_name]
        losses['hybrid'] = loss


    losses = dict((k, v.item()) for k, v in losses.items())
    return loss, losses
