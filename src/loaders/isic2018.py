from datasets.isic2018 import ISIC2018DatasetFast
from torch.utils.data import DataLoader, Subset
from modules.transforms import DiffusionTransform, DataAugmentationTransform
import albumentations as A
import glob



def get_isic2018(config, logger=None, verbose=False):

    if logger: print = logger.info

    INPUT_SIZE = config["dataset"]["input_size"]
    DT = DiffusionTransform((INPUT_SIZE, INPUT_SIZE))
    AUGT = DataAugmentationTransform((INPUT_SIZE, INPUT_SIZE))
    
    img_dir = "ISIC2018_Task1-2_Training_Input"
    msk_dir = "ISIC2018_Task1_Training_GroundTruth"
    img_path_list = glob.glob(f"{config['dataset']['data_dir']}/{img_dir}/ISIC*.jpg")
    
    pixel_level_transform = AUGT.get_pixel_level_transform(config["augmentation"], img_path_list=img_path_list)
    spacial_level_transform = AUGT.get_spacial_level_transform(config["augmentation"])
    tr_aug_transform = A.Compose([
        A.Compose(pixel_level_transform, p=config["augmentation"]["levels"]["pixel"]["p"]), 
        A.Compose(spacial_level_transform, p=config["augmentation"]["levels"]["spacial"]["p"])
    ], p=config["augmentation"]["p"])

    # ----------------- dataset --------------------
    if config["dataset"]["class_name"] == "ISIC2018Dataset":

        dataset = ISIC2018Dataset(
            data_dir=config["dataset"]["data_dir"],
            one_hot=False,
            # aug=AUGT.get_aug_policy_3(),
            # transform=AUGT.get_spatial_transform(),
            img_transform=DT.get_forward_transform_img(),
            msk_transform=DT.get_forward_transform_msk(),
            logger=logger
        )

        #         tr_dataset = Subset(dataset, range(0    , 38       ))
        #         vl_dataset = Subset(dataset, range(38   , 38+25    ))
        #         te_dataset = Subset(dataset, range(38+25, len(dataset)))
        tr_dataset = Subset(dataset, range(0, 1815))
        vl_dataset = Subset(dataset, range(1815, 1815 + 259))
        te_dataset = Subset(dataset, range(1815 + 259, 1815 + 259 + 520))
        # We consider 1815 samples for training, 259 samples for validation and 520 samples for testing
        # !cat ~/deeplearning/skin/Prepare_ISIC2018.py

    elif config["dataset"]["class_name"] == "ISIC2018DatasetFast":
        # preparing training dataset
        tr_dataset = ISIC2018DatasetFast(
            mode="tr",
            data_dir=config["dataset"]["data_dir"],
            one_hot=False,
            image_size=config["dataset"]["input_size"],
            aug=tr_aug_transform,
            # transform=AUGT.get_spatial_transform(),
            img_transform=DT.get_forward_transform_img(),
            msk_transform=DT.get_forward_transform_msk(),
            add_boundary_mask=config["dataset"]["add_boundary_mask"],
            add_boundary_dist=config["dataset"]["add_boundary_dist"],
            logger=logger,
            data_scale=config["dataset"]["data_scale"]
        )
        vl_dataset = ISIC2018DatasetFast(
            mode="vl",
            data_dir=config["dataset"]["data_dir"],
            one_hot=False,
            image_size=config["dataset"]["input_size"],
            # aug_empty=AUGT.get_val_test(),
            # transform=AUGT.get_spatial_transform(),
            img_transform=DT.get_forward_transform_img(),
            msk_transform=DT.get_forward_transform_msk(),
            add_boundary_mask=config["dataset"]["add_boundary_mask"],
            add_boundary_dist=config["dataset"]["add_boundary_dist"],
            logger=logger,
            data_scale=config["dataset"]["data_scale"]
        )
        te_dataset = ISIC2018DatasetFast(
            mode="te",
            data_dir=config["dataset"]["data_dir"],
            one_hot=False,
            image_size=config["dataset"]["input_size"],
            # aug_empty=AUGT.get_val_test(),
            # transform=AUGT.get_spatial_transform(),
            img_transform=DT.get_forward_transform_img(),
            msk_transform=DT.get_forward_transform_msk(),
            add_boundary_mask=config["dataset"]["add_boundary_mask"],
            add_boundary_dist=config["dataset"]["add_boundary_dist"],
            logger=logger,
            data_scale=config["dataset"]["data_scale"]
        )

    else:
        message = "In the config file, `dataset>class_name` should be in: ['ISIC2018Dataset', 'ISIC2018DatasetFast']"
        if logger: 
            logger.exception(message)
        else:
            raise ValueError(message)

    if verbose:
        print("ISIC 2018:")
        print(f"├──> Length of trainig_dataset:\t   {len(tr_dataset)}")
        print(f"├──> Length of validation_dataset: {len(vl_dataset)}")
        print(f"└──> Length of test_dataset:\t   {len(te_dataset)}")

    # prepare train dataloader
    tr_dataloader = DataLoader(tr_dataset, **config["data_loader"]["train"])

    # prepare validation dataloader
    vl_dataloader = DataLoader(vl_dataset, **config["data_loader"]["validation"])

    # prepare test dataloader
    te_dataloader = DataLoader(te_dataset, **config["data_loader"]["test"])

    return {
        "tr": {"dataset": tr_dataset, "loader": tr_dataloader},
        "vl": {"dataset": vl_dataset, "loader": vl_dataloader},
        "te": {"dataset": te_dataset, "loader": te_dataloader},
    }


# # test and visualize the input data
# from utils.helper_funcs import show_sbs
# for sample in tr_dataloader:
#     img = sample['image']
#     msk = sample['mask']
#     print("Training")
#     print(img.shape, msk.shape)
#     show_sbs(img[0], msk[0])
#     break

# for sample in vl_dataloader:
#     img = sample['image']
#     msk = sample['mask']
#     print("Validation")
#     show_sbs(img[0], msk[0])
#     break

# for sample in te_dataloader:
#     img = sample['image']
#     msk = sample['mask']
#     print("Test")
#     show_sbs(img[0], msk[0])
#     break
