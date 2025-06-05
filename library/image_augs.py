################################################################################
# image_augs.py
# Written by Jacob Wheelock & Erin Shappell for Lu Lab
# 
# This module defines data augmentation pipelines using the Albumentations library
# for training and validation datasets in object detection tasks.
#
################################################################################
# Imports
import albumentations as A
from albumentations.pytorch import ToTensorV2

################################################################################    
def get_train_transform():
    """
    Returns a composition of data augmentation transforms for training images and bounding boxes.

    Inputs:
        None

    Output:
        albumentations.Compose: A composition of image transformations applied with specified probabilities,
                                including flipping, rotation, blurring, and tensor conversion, with bounding
                                box support in Pascal VOC format.
    """
    return A.Compose([
        A.Flip(0.5),
        A.RandomRotate90(0.5),
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

################################################################################    
def get_valid_transform():
    """
    Returns the transformation applied to validation images and bounding boxes.

    Inputs:
        None

    Output:
        albumentations.Compose: A composition that converts images and bounding boxes to tensors,
                                with bounding box support in Pascal VOC format.
    """
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc', 
        'label_fields': ['labels']
    })
