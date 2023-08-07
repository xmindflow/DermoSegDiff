# DermoSegDiff: A Boundary-aware Segmentation Diffusion Model for Skin Lesion Delineation <br> <span style="float: rigth"><sub><sup>$\text{(\textcolor{teal}{MICCAI 2023 PRIME Workshop})}$</sub></sup></span>

Skin lesion segmentation plays a critical role in the early detection and accurate diagnosis of dermatological conditions. Denoising Diffusion Probabilistic Models (DDPMs) have recently gained attention for their exceptional image-generation capabilities. Building on these advancements, we propose DermoSegDiff, a novel framework for skin lesion segmentation that incorporates boundary information during the learning process. Our approach introduces a novel loss function that prioritizes the boundaries during training, gradually reducing the significance of other regions. We also introduce a novel U-Net-based denoising network that proficiently integrates noise and semantic information inside the network. Experimental results on multiple skin segmentation datasets demonstrate the superiority of DermoSegDiff over existing CNN, transformer, and diffusion-based approaches, showcasing its effectiveness and generalization in various scenarios.
<p align="center">
  <em>Network</em><br/>
  <img width="600" alt="image" src="https://github.com/mindflow-institue/DermoSegDiff/assets/6207884/7619985e-d894-4ada-9125-9f40a32bae7d">
  <br/>
  <br/>
  <em>Method</em></br>
  <img width="800" alt="image" src="https://github.com/mindflow-institue/DermoSegDiff/assets/61879630/0919e613-972a-47ac-ac79-04a2ae51ed1e">
</p>

## Citation
```bibtex
@inproceedings{bozorgpour2023dermosegdiff,
  title={DermoSegDiff: A Boundary-aware Segmentation Diffusion Model for Skin Lesion Delineation},
  author={Bozorgpour, Afshin and Sadegheih, Yousef and Kazerouni, Amirhossein and Azad, Reza and Merhof, Dorit},
  booktitle={International Workshop on PRedictive Intelligence in MEdicine},
  pages={--},
  year={2023}.
  organization={Springer}
}
```
<p align="center">
  <img width="620" alt="image" src="https://github.com/mindflow-institue/DermoSegDiff/assets/6207884/30bb1483-e9f8-44df-bede-13238df6f4f0">
</p>

## News
- July 25, 2023: Accepted in MICCAI 2023 PRIME Workshop! 🥳

## How to use

  ### Requirements
  
  - Ubuntu 16.04 or higher
  - CUDA 11.1 or higher
  - Python v3.7 or higher
  - Pytorch v1.7 or higher
  - Hardware Spec
    - GPU with 12GB memory or larger capacity (With low GPU memory you need to change and decrease `dim_x_mults`, `dim_g_mults`, `dim_x`, and `dim_g` params. You also need to change `batch_size` respectively. If you tune it well you won't lose considerable capability!)
    - _For our experiments, we used 1GPU(A100-80G)_

  
  ```albumentations==1.3.1
  einops==0.6.1
  ema_pytorch==0.2.3
  matplotlib==3.7.2
  numpy==1.24.4
  opencv==4.6.0
  opencv_python_headless==4.8.0.74
  Pillow==10.0.0
  PyYAML==6.0.1
  scikit_image==0.19.3
  scipy==1.6.3
  termcolor==2.3.0
  torch==2.0.1
  torchvision==0.15.2
  tqdm==4.65.0
  ```
  `pip install -r requirements.txt`

  ### Model weights
  You can download the learned weights in the following.
   Dataset   | Model          | download link 
  -----------|----------------|----------------
   ISIC2018  | DermoSegDiff-A | [[Download](https://uniregensburg-my.sharepoint.com/:f:/g/personal/say26747_ads_uni-regensburg_de/EhsfBqr1Z-lCr6KaOkRM3EgBIVTv8ew2rEvMWpFFOPOi1w?e=ifo9jF)] 
   PH2       | DermoSegDiff-B | [[Download](https://uniregensburg-my.sharepoint.com/:f:/g/personal/say26747_ads_uni-regensburg_de/EoCkyNc5yeRFtD-KTFbF0gcB8lbjMLY6t1D7tMYq7yTkfw?e=tfGHee)] 
  
  ### Training
  For the training stage you need to choose the relevant config file and modify it by setting the required directories and changing variables if it's desired, and from inside the `src` folder run the following command by pathing the prepared config file:
  
  ```python src/training.py -c /path/to/config/file```

  You can also overload some parameters while running the above command:

  ```bash
usage: [-h] -c CONFIG_FILE [-n MODEL_NAME] [-s INPUT_SIZE] [-b BATCH_SIZE] [-l LEARNING_RATE]
         [-t TIMESTEPS] [-S {linear,quadratic,cosine,sigmoid}] [-e EPOCHS] [--beta_start BETA_START]
         [--beta_end BETA_END] [-D [MODEL_DIM_MULTS [MODEL_DIM_MULTS ...]]] [-E ENSEMBLE]
         [--model_dim_x MODEL_DIM_X] [--model_dim_g MODEL_DIM_G]
         [--model_dim_x_mults [MODEL_DIM_X_MULTS [MODEL_DIM_X_MULTS ...]]]
         [--model_dim_g_mults [MODEL_DIM_G_MULTS [MODEL_DIM_G_MULTS ...]]]
         [--training_optimizer_betas [TRAINING_OPTIMIZER_BETAS [TRAINING_OPTIMIZER_BETAS ...]]]
         [--training_scheduler_factor TRAINING_SCHEDULER_FACTOR]
         [--training_scheduler_patience TRAINING_SCHEDULER_PATIENCE] [--augmentation_p AUGMENTATION_P]
 
  ```
  
  ### Sampling & Test
  For sampling and testing, you need to pass a relevant config file as same as training:
  
  ```python src/testing.py -c /path/to/config/file```
  
  To run with arbitrary weigths you need to change `testing -> model_weigths -> overload` to `true` and write the desired weights path in `testing -> model_weigths -> file_path`.
  
  ### Evaluation
  
  <p align="center">
    <img width="800" alt="image" src="https://github.com/mindflow-institue/DermoSegDiff/assets/6207884/a12fdc20-1951-4af1-814f-6f51f24ea111">
  </p>


## References
- https://github.com/lucidrains/denoising-diffusion-pytorch

