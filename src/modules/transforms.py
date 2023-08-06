import numpy as np
from torchvision import transforms as T
import albumentations as A
from albumentations import *



class DataAugmentationTransform(object):
    def __init__(self, input_size):
        self.input_size = input_size

    def get_aug_policy_1(self):
        dns_1 = A.Compose([
            A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightness(limit=0.1, p=0.75),
            A.RandomContrast(limit=0.1, p=0.75),
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),
                A.GaussNoise(var_limit=(5.0, 30.0)),
            ], p=0.7),

            A.OneOf([
                A.OpticalDistortion(distort_limit=1.0),
                A.GridDistortion(num_steps=5, distort_limit=1.),
                A.ElasticTransform(alpha=3),
                # A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
            ], p=0.7),

            A.CLAHE(clip_limit=4.0, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            A.Resize(self.input_size[0], self.input_size[1]),
            A.Cutout(max_h_size=int(self.input_size[0] * 0.2), max_w_size=int(self.input_size[1] * 0.2), num_holes=1, p=0.7),    
            A.Normalize(),
        ])
        return dns_1

    def get_aug_policy_2(self):
        dns_1 = A.Compose([
            A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightness(limit=0.1, p=0.75),
            A.RandomContrast(limit=0.1, p=0.75),
            A.OneOf([
                A.MotionBlur(blur_limit=3),
                A.MedianBlur(blur_limit=3),
                A.GaussianBlur(blur_limit=1),
                A.GaussNoise(var_limit=(5.0, 30.0)),
            ], p=0.7),

            A.OneOf([
                A.OpticalDistortion(distort_limit=1.0),
                A.GridDistortion(num_steps=3, distort_limit=1.),
                A.ElasticTransform(alpha=3),
                # A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
            ], p=0.7),

            A.CLAHE(clip_limit=3.0, p=0.7),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            A.Resize(self.input_size[0], self.input_size[1]),
            A.Cutout(max_h_size=int(self.input_size[0] * 0.1), max_w_size=int(self.input_size[1] * 0.1), num_holes=3, p=0.7),    
            A.Normalize(),
        ])
        return dns_1

    def get_aug_policy_3(self):
        dns_1 = A.Compose([
            A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightness(limit=0.1, p=0.75),
            A.RandomContrast(limit=0.1, p=0.75),
            A.OneOf([
                # A.MotionBlur(blur_limit=3),
                # A.MedianBlur(blur_limit=3),
                # A.GaussianBlur(blur_limit=1),
                A.GaussNoise(var_limit=(1.0, 5.0)),
            ], p=0.7),

            A.OneOf([
                A.OpticalDistortion(distort_limit=.1),
                A.GridDistortion(num_steps=1, distort_limit=1.),
                # A.ElasticTransform(alpha=200),
                # A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
            ], p=0.7),

            # A.CLAHE(clip_limit=3.0, p=0.7),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            A.Resize(self.input_size[0], self.input_size[1]),
            
            A.OneOf([
                A.Cutout(max_h_size=int(self.input_size[0] * 0.375), max_w_size=int(self.input_size[1] * 0.375), num_holes=1, p=0.7),    
                A.Cutout(max_h_size=int(self.input_size[0] * 0.200), max_w_size=int(self.input_size[1] * 0.200), num_holes=2, p=0.7),    
                A.Cutout(max_h_size=int(self.input_size[0] * 0.100), max_w_size=int(self.input_size[1] * 0.100), num_holes=3, p=0.7),    
            ], p=0.7),
            
            A.Normalize(),
        ])
        return dns_1


    def get_pixel_level_transform(self, config, img_path_list=None):
        pixel_transform = A.Compose([globals()[t](**p) for t, p in config['levels']['pixel']['transforms'].items()])
        return pixel_transform
        
    def get_spacial_level_transform(self, config, img_path_list=None):
        spacial_transform = A.Compose([globals()[t](**p) for t, p in config['levels']['spacial']['transforms'].items()])
        return spacial_transform

    
    def get_val_test(self):
        dns_1 = A.Compose([
            # A.CLAHE(clip_limit=3.0, p=1),
            A.Resize(self.input_size[0], self.input_size[1]),
            A.Normalize(),
        ])
        return dns_1

    def get_geo_transform(self):
        sim_geo_transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomApply([T.RandomRotation(degrees=(0, 180)),], 0.25),
        ])
        pro_geo_transforms = T.Compose([
            T.RandomApply([
                T.RandomAffine(
                    degrees=(30, 70), translate=(0.01, 0.1), 
                    scale=(0.8, 1.2), shear=(0, 0.2)),
            ], 0.5),
            T.RandomPerspective(distortion_scale=0.2, p=0.5),
        ])
        geo_transform = T.RandomChoice([
            T.RandomApply([sim_geo_transforms], p=0.6),
            T.RandomApply([pro_geo_transforms], p=0.4)
        ])
        return geo_transform
    def get_color_transform(self):
        color_transform = T.ColorJitter(brightness=0.3, hue=0.05, saturation=0.1, contrast=0.3)
        return color_transform
    
    def get_other_transform(self):
        pad_transform = T.Compose([
            T.RandomChoice([
                T.Pad(padding=1), T.Pad(padding=2), T.Pad(padding=3),
                T.Pad(padding=4), T.Pad(padding=5), T.Pad(padding=6),
                T.Pad(padding=7), T.Pad(padding=8), T.Pad(padding=9),
            ]),
            # T.Resize(size=self.input_size)
        ])
        other_transform = T.RandomChoice([
            T.RandomApply([pad_transform], p=0.8)
        ])
        return other_transform
        
    def get_blur_transform(self):
        gaussian_transforms = T.RandomChoice([
            T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 1)),
            T.GaussianBlur(kernel_size=(5, 5), sigma=(1, 5)),
        ])
        elastic_transforms = T.RandomChoice([
            T.ElasticTransform(alpha=1.0, sigma=2.,),
            T.ElasticTransform(alpha=2.0, sigma=5.,),
            # T.ElasticTransform(alpha=250.0, sigma=4.0,),
            # T.ElasticTransform(alpha=50.0, sigma=4.0,),
            # T.ElasticTransform(alpha=150.0, sigma=2.0,),
            # T.ElasticTransform(alpha=250.0, sigma=1.0,),
        ])
        blur_transform = T.RandomChoice([
            T.RandomApply([gaussian_transforms,], p=0.6),
            T.RandomApply([elastic_transforms,], p=0.2),
            T.RandomApply([T.AugMix()], p=0.1),
        ])
        return blur_transform   
    
    def get_enhancement_transform(self):
        enhancement_transform = T.Compose([
            T.RandomAutocontrast(p=0.3),
            T.RandomEqualize(p=0.3), 
            T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        ])
        return enhancement_transform
    
    def get_spatial_transform(self):
        geo_transform = self.get_geo_transform()
        other_transform = self.get_other_transform()
        spatial_augmentation_transform = T.Compose([
            T.ToPILImage(),
            T.RandomApply([geo_transform,], p=0.3),
            T.RandomApply([other_transform,], p=0.2),
            T.ToTensor()
        ]) 
        return spatial_augmentation_transform
    
    def get_contextual_transform(self):
        color_transform = self.get_color_transform()
        blur_transform = self.get_blur_transform()
        enhancement_transform = self.get_enhancement_transform()
        posterize_transform = T.RandomPosterize(bits=2)
        
        augmentation_transform_img = T.Compose([
            T.ToPILImage(),
            T.RandomApply([color_transform,], p=0.2),
            T.RandomApply([blur_transform,], p=0.1),
            T.RandomApply([enhancement_transform,], p=0.2),
            T.RandomApply([posterize_transform], p=0.15),
            T.ToTensor()
        ])
        
        return augmentation_transform_img


# =============================================================================

class DiffusionTransform(object):
    def __init__(self, input_size) -> None:
        self.input_size = input_size
        self.aug_transforms_obj = DataAugmentationTransform(input_size)
        
        self.contextual_transform = self.aug_transforms_obj.get_contextual_transform()        
        self.spatial_transform = self.aug_transforms_obj.get_spatial_transform()
        
    def get_forward_transform_img(self):
        return T.Compose(
            [
                # T.RandomApply([self.contextual_transform], p=.25),
                T.Resize(self.input_size),
                T.Lambda(lambda t: (t - t.min()) / (t.max() - t.min())),
                # Lambda(lambda t: (t * 2) - 1),
            ]
        )

    def get_forward_transform_msk(self):
        return T.Compose(
            [
                T.Resize(self.input_size, interpolation=T.InterpolationMode.NEAREST),
                T.Lambda(lambda t: (t - t.min()) / (t.max() - t.min())),
                T.Lambda(lambda t: (t * 2) - 1),
            ]
        )

    def get_reverse_transform_to_numpy(self):
        return T.Compose(
            [
                T.Lambda(lambda t: (t + 1) / 2),
                T.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
                T.Lambda(lambda t: t.detach().numpy().astype(np.float32)),
            ]
        )

    def get_reverse_transform_to_pil(self):
        return T.Compose(
            [
                T.Lambda(lambda t: (t + 1) / 2),
                T.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
                T.Lambda(lambda t: t * 255.0),
                T.Lambda(lambda t: t.detach().numpy().astype(np.uint8)),
                T.ToPILImage(),
            ]
        )

    def get_reverse_transform_to_binary(self, thr=0.5):
        return T.Compose(
            [
                T.Lambda(lambda t: (t + 1) / 2),
                T.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
                T.Lambda(lambda t: (t > thr).to(t.dtype)),
                T.Lambda(lambda t: t.detach().numpy().astype(np.uint8)),
            ]
        )

