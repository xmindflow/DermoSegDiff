import argparse



def get_argparser():
  parser = argparse.ArgumentParser(
    prog='DermoSegDiff',
    description='DermoSegDiff: A Boundary-aware Segmentation Diffusion Model for Skin Lesion Delineation',
    epilog=''
  )

  parser.add_argument('-c', '--config_file', type=str, required=True, help="")
  parser.add_argument('-n', '--model_name', type=str, help="")
  parser.add_argument('-s', '--input_size', type=int, help="")
  parser.add_argument('-b', '--batch_size', type=int, help="")
  parser.add_argument('-l', '--learning_rate', type=float, help="")
  parser.add_argument('-t', '--timesteps', type=int, help="")
  parser.add_argument('-S', '--ds_mode', type=str, choices=['linear', 'quadratic', 'cosine', 'sigmoid'], help="linear, quadratic, cosine, sigmoid")
  parser.add_argument('-e', '--epochs', type=int, help="")
  parser.add_argument(      '--beta_start', type=float, help="")
  parser.add_argument(      '--beta_end', type=float, help="")
  parser.add_argument('-D', '--model_dim_mults', type=int, nargs='*', help="1 2 4")
  parser.add_argument('-E', '--ensemble', type=int, help="")
  
  parser.add_argument('--model_dim_x', type=int, help="128")
  parser.add_argument('--model_dim_g', type=int, help="64")
  parser.add_argument('--model_dim_x_mults', type=int, nargs="*", help="1 2 3 4 05 06")
  parser.add_argument('--model_dim_g_mults', type=int, nargs="*", help="1 2 4 8 16 32")
  parser.add_argument('--training_optimizer_betas', type=float, nargs="*", help="0.7 0.98")
  parser.add_argument('--training_scheduler_factor', type=float, help="0.5")
  parser.add_argument('--training_scheduler_patience', type=int, help="5")
  parser.add_argument('--augmentation_p', type=float, help="0.5")
  
  parser.add_argument('-v', '--verbose', action='store_true')
  
  return parser



def sync_config(config, args):

  if args.model_name:
    config["model"]["name"] = args.model_name
  if args.input_size:
    config["dataset"]["input_size"] = args.input_size
  if args.batch_size:
    config["data_loader"]["train"]["batch_size"] = args.batch_size
  if args.learning_rate:
    config["training"]["optimizer"]["params"]["lr"] = args.learning_rate
  if args.timesteps:
    config["diffusion"]["schedule"]["timesteps"] = args.timesteps
  if args.ds_mode:
    config["diffusion"]["schedule"]["mode"] = args.ds_mode
  if args.epochs:
    config["training"]["epochs"] = args.epochs
  if args.beta_start:
    config["diffusion"]["schedule"]["beta_start"] = args.beta_start
  if args.beta_end:
    config["diffusion"]["schedule"]["beta_end"] = args.beta_end
  if args.model_dim_mults:
    config["model"]["params"]["dim_mults"] = args.model_dim_mults
  if args.ensemble:
    config["testing"]["ensemble"] = args.ensemble
    
  if args.model_dim_x:
    config["model"]["params"]["dim_x"] = args.model_dim_x
  if args.model_dim_g:
    config["model"]["params"]["dim_g"] = args.model_dim_g
  if args.model_dim_x_mults:
    config["model"]["params"]["dim_x_mults"] = args.model_dim_x_mults
  if args.model_dim_g_mults:
    config["model"]["params"]["dim_g_mults"] = args.model_dim_g_mults
  if args.training_optimizer_betas:
    config["training"]["optimizer"]["params"]["betas"] = args.training_optimizer_betas
  if args.training_scheduler_factor:
    config["training"]["scheduler"]["factor"] = args.training_scheduler_factor
  if args.training_scheduler_patience:
    config["training"]["scheduler"]["patience"] = args.training_scheduler_patience
  if args.augmentation_p:
    config["augmentation"]["p"] = args.augmentation_p

    
# --------- default values ------------
  if not "data_scale" in config["dataset"].keys():
    config["dataset"]["data_scale"] = "full"
    
  if not "model_weigths" in config["testing"]: 
    config["testing"]["model_weigths"] = {"overload": False, "file_path": ""}
  
  if not "intial_weights" in config["training"]:
    config["training"]["intial_weights"] = {"use": False, "file_path": ""}
    
    
# --------- add auxiliary params ----------
  loss = config["training"]["loss"]
  if len(loss.keys()) == 1:
    config['training']['loss_name'] = list(loss.keys())[0]
  elif len(loss.keys()) > 1:
    config['training']['loss_name'] = "hybrid"
  else:
    raise ValueError("you must determine the <loss> parameter on the <training> section inside config file.")
    
  return config
  
