# DermoSegDiff: A Boundary-aware Segmentation Diffusion Model for Skin Lesion Delineation - MICCAI 2023 PRIME Workshop

Skin lesion segmentation plays a critical role in the early detection and accurate diagnosis of dermatological conditions. Denoising Diffusion Probabilistic Models (DDPMs) have recently gained attention for their exceptional image-generation capabilities. Building on these advancements, we propose DermoSegDiff, a novel framework for skin lesion segmentation that incorporates boundary information during the learning process. Our approach introduces a novel loss function that prioritizes the boundaries during training, gradually reducing the significance of other regions. We also introduce a novel U-Net-based denoising network that proficiently integrates noise and semantic information inside the network. Experimental results on multiple skin segmentation datasets demonstrate the superiority of DermoSegDiff over existing CNN, transformer, and diffusion-based approaches, showcasing its effectiveness and generalization in various scenarios.

![method](https://github.com/mindflow-institue/DermoSegDiff/assets/61879630/0919e613-972a-47ac-ac79-04a2ae51ed1e)

## Citation
```python
@inproceedings{bozorgpour2023dermosegdiff,
  title={DermoSegDiff: A Boundary-aware Segmentation Diffusion Model for Skin Lesion Delineation},
  author={Bozorgpour, Afshin and Sadegheih, Yousef and Kazerouni, Amirhossein and Azad, Reza and Merhof, Dorit},
  booktitle={International Workshop on PRedictive Intelligence In MEdicine},
  pages={--},
  year={2023}.
  organization={Springer}
}
```

## News
- July 25, 2023: Accepted in MICCAI 2023 PRIME Workshop! 🥳

## How to use

  ### Requirements
  
`pip install -r requirements.txt`

  ### Model weights

You can download the learned weights in the following table. 

Model | Dataset |Learned weights
------------ | -------------|----
Baseline | [ISIC 2018]() | [Download]()
DermoSegDiff-A | [ISIC 2018]() | [Download]()
DermoSegDiff-B | [ISIC 2018]() | [Download]()

  ### Training

  ### Sampling

  ### Evaluation

## References
-
-
