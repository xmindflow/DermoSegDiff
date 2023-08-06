from pathlib import Path
import numpy as np
import random
import torch
from torch.optim import Adam, SGD, AdamW
from utils.helper_funcs import (
    load_config,
    get_model_path,
    get_conf_name,
    print_config,
)
from models import *
from forward.forward_schedules import ForwardSchedule
from forward.forward_process import ForwardProcess
from torch.optim import lr_scheduler
from train_validate import train, validate
from loaders.dataloaders import get_dataloaders
from torch.utils.tensorboard import SummaryWriter
import sys, os
from common.logging import get_logger
from argument import get_argparser, sync_config
import warnings
warnings.filterwarnings('ignore')



# ------------------- params --------------------
argparser = get_argparser()
args = argparser.parse_args(sys.argv[1:])

config = load_config(args.config_file)
config = sync_config(config, args)

logger = get_logger(filename=f"{config['model']['name']}", dir=f"logs/{config['dataset']['name']}")
print_config(config, logger)

# create the writer for tensorboard
writer = SummaryWriter(f'{config["run"]["writer_dir"]}/{config["model"]["name"]}')

# variables
timesteps = config["diffusion"]["schedule"]["timesteps"]
epochs = config["training"]["epochs"]
input_size = config["dataset"]["input_size"]
batch_size = config["data_loader"]["train"]["batch_size"]
img_channels = config["dataset"]["img_channels"]
msk_channels = config["dataset"]["msk_channels"]
ID = get_conf_name(config)

# device
device = torch.device(config["run"]["device"])
logger.info(f"Device is <{device}>")


start_epoch = 0
best_vl_loss = np.Inf
best_vl_losses = {}
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed()


# --------- check required dirs --------------------
Path(config["model"]["save_dir"]).mkdir(exist_ok=True, parents=True)


forward_schedule = ForwardSchedule(**config["diffusion"]["schedule"])
forward_process = ForwardProcess(forward_schedule)

# --------------- Datasets and Dataloaders -----------------
tr_dataloader, vl_dataloader = get_dataloaders(config, ["tr", "vl"])


Net = globals()[config["model"]["class"]]
model = Net(**config["model"]["params"])

# writer.add_graph(model, (torch.randn(1, img_channels+msk_channels, input_size, input_size), torch.tensor([1])))
# writer.add_graph(sample, (forward_schedule, model, batch["image"], 1))
model = model.to(device)
total_params = sum(p.numel() for p in model.parameters())
logger.info(f"Number of model parameters: {total_params}")


tr_prms = config["training"]
optimizer = globals()[tr_prms["optimizer"]["name"]](
    model.parameters(), **tr_prms["optimizer"]["params"]
)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "min", **tr_prms["scheduler"])
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, last_epoch=epochs, verbose=True)
# step_lr_schedule = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# ------------------------ EMA -------------------------------
# https://github.com/lucidrains/ema-pytorch
from ema_pytorch import EMA

try:
    if config["training"]["ema"]["use"]:
        ema = EMA(model=model, **config["training"]["ema"]["params"])
        ema.to(device)
    else:
        ema = None
except KeyError:
    logger.exception("You need to determine the EMA parameters at <config.training>!")


if config["run"]["continue_training"] or config["training"]["intial_weights"]["use"]:
    if config["run"]["continue_training"]:
        model_path = get_model_path(name=ID, dir=config["model"]["save_dir"])
    else:
        model_path = config["training"]["intial_weights"]["file_path"]
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        if config["run"]["continue_training"]:
            if checkpoint["epochs"] > checkpoint["epoch"] + 1:
                best_vl_loss = checkpoint["vl_loss"]
                model.load_state_dict(checkpoint["model"])
                start_epoch = checkpoint["epoch"] + 1
                optimizer.load_state_dict(checkpoint["optimizer"])
                if ema:
                    ema.load_state_dict(checkpoint["ema"])

                logger.info(f"Loaded the model state (ep:{checkpoint['epoch']+1}/{checkpoint['epochs']}) to continue training from the following path:")
                logger.info(f" -> {model_path}\n")
            else:
                logger.warning("the net already trained!")
                sys.exit()
        else:
            model.load_state_dict(checkpoint["model"])
            if ema:
                ema = EMA(model=model, **config["training"]["ema"]["params"])
                ema.to(device)
            logger.info(f"Loaded initial weights")
            logger.info(f" -> {model_path}\n")
            
    except:
        logger.warning("There is a problem with loading the previous model to continue training.")
        logger.warning(" --> Do you want to train the model from the beginning? (y/N):")
        user_decision = input()
        if (user_decision != "y"):
            exit()
else:
    model_path = get_model_path(name=ID, dir=config["model"]["save_dir"])
    if os.path.isfile(model_path):
        logger.warning(f"There is a model weights at determind directory with desired name: {ID}")
        logger.warning(" --> Do you want to train the model from the beginning? It will overwrite the current weights! (y/N):")
        user_decision = input()
        if (user_decision != "y"):
            exit()


for epoch in range(start_epoch, epochs):

    tr_losses, model = train(
        model,
        tr_dataloader,
        forward_process,
        device,
        optimizer,
        ema=ema,
        cfg=config,
        extra={"skip_steps": 10, "prefix": f"ep:{epoch+1}/{epochs}"},
        logger=logger
    )

    vl_losses = validate(
        ema.ema_model if ema else model,
        vl_dataloader,
        forward_process,
        device,
        cfg=config,
        vl_runs=3,
        logger=logger
    )

    tr_loss = np.mean([l[0] for l in tr_losses])
    vl_loss = np.mean([l[0] for l in vl_losses])

    writer.add_scalars(
        f"Loss/train vs validation/{config['training']['loss_name']}",
        {"Train": tr_loss, "Validation": vl_loss},
        epoch,
    )

    # ---------- tr losses -------------
    lns = tr_losses[0][1].keys()
    tr_losses_dict = dict((n, np.mean([d[1][n] for d in tr_losses])) for n in lns)
    
    # ---------- vl losses -------------
    assert lns == vl_losses[0][1].keys(), "Ops... reported losses are different between tr and vl!"
    vl_losses_dict = dict((n, np.mean([d[1][n] for d in vl_losses])) for n in lns)

    # --------- add tr, vl scalars (all losses) ------
    for ln, v in tr_losses_dict.items():
        writer.add_scalars(
            f"Losses/{ln.upper()}",
            {"train": v, "validation": vl_losses_dict[ln]},
            epoch,
        )

    extra_tr_losses_txt = ", ".join(
        [f"{ln}: {v:0.6f}" for ln, v in tr_losses_dict.items()]
    )
    extra_vl_losses_txt = ", ".join(
        [f"{ln}: {v:0.6f}" for ln, v in vl_losses_dict.items()]
    )
    extra_losses_txt = (
        f"  >> tr -> {extra_tr_losses_txt}\n  >> vl -> {extra_vl_losses_txt}\n"
    )
    logger.info(
        f"\nep:{epoch+1:03d}/{epochs:03d}, tr_loss: {tr_loss:0.8f}, vl_loss: {vl_loss:0.8f}\n{extra_losses_txt}"
    )
    
    if best_vl_loss > vl_loss:
        logger.info(
            f">>> Found a better model: last-vl-loss:{best_vl_loss:0.8f}, new-vl-loss:{vl_loss:0.8f}"
        )
        best_vl_loss = vl_loss
        model_path = get_model_path(name=ID, dir=config["model"]["save_dir"])
        # torch.save(model.state_dict(), model_path)

        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
            "epochs": epochs,
            "optimizer": optimizer.state_dict(),
            "ema": ema.state_dict() if ema else None,    
            "vl_loss": vl_loss,
        }

        torch.save(checkpoint, model_path)


writer.flush()
writer.close()
