################################################################################
# plotting.py
# Written by Jacob Wheelock & Erin Shappell for Lu Lab
# 
# This module provides a helper function to visualize annotated training samples
# from an object detection dataset. Useful for debugging datasets, inspecting annotations, 
# and verifying preprocessing steps.
#
################################################################################
# Imports
import cv2
import matplotlib.pyplot as plt
from .utils import getDataset

################################################################################   
def visualize_sample(TRAIN_DIR, RESIZE_TO, index):
    """
    Visualizes a single sample image with annotated bounding boxes and class labels from a dataset.

    Inputs:
        TRAIN_DIR (str): Directory path containing training images and annotations.
        RESIZE_TO (int): Size (width and height) to which images are resized.
        index (int):     Index of the sample image in the dataset to visualize.

    Outputs:
        matplotlib.figure.Figure: Figure object displaying the image with bounding boxes and labels.

    """
    dataset = getDataset(
        TRAIN_DIR, RESIZE_TO, RESIZE_TO
    )
    
    image, target = dataset[index]
    fig = plt.figure(figsize=(4, 4))
    ax = fig.subplots()
    ax.axis('off')  # This hides both the axes and the ticks
    plt.tight_layout()  # This automatically adjusts the padding

    for i in range(0,len(target['boxes'])):
        box = target['boxes'][i]
        classesBG = dataset.classes
        label = classesBG[target['labels'][i].item()]
    
        cv2.rectangle(
            image, 
            (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
            (0, 255, 0), 1
            )
        cv2.putText(
            image, label, (int(box[0]), int(box[1]-5)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
    plt.imshow(image)
    return fig