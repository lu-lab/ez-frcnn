################################################################################
# training.py
# Written by Jacob Wheelock & Erin Shappell for Lu Lab
# 
# This module provides functions to create, load, train, and validate a Faster R-CNN model
# for object detection tasks using PyTorch and torchvision.
#
################################################################################
# Imports
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
#from tqdm.auto import tqdm
import time
from torch.utils.data import Subset, DataLoader
from torchvision.ops import box_iou
import numpy as np
from .utils import get_loaders
from .utils import collate_fn

################################################################################   
def create_model(num_classes):
    """
    Creates a Faster R-CNN model pre-trained on COCO and modifies its head for a custom number of classes.

    Inputs:
        num_classes (int): Number of output classes for detection (including background).

    Outputs:
        torchvision.models.detection.FasterRCNN: The modified Faster R-CNN model ready for training or inference.
    """
    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='COCO_V1')
    
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model

################################################################################   
def load_model_train(model_name, MODEL_DIR, NUM_CLASSES):
    """
    Loads a trained model for inference or further training.

    Inputs:
        model_name (str):  Filename of the saved model weights.
        MODEL_DIR (str):   Directory where the model weights are stored.
        NUM_CLASSES (int): Number of output classes the model predicts.

    Outputs:
        torch.nn.Module: The model loaded with trained weights, moved to the appropriate device (CPU or GPU).
    """
    # set the computation device
    modelPath = './models/' + model_name
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # load the model and the trained weights
    model = create_model(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(
        modelPath, map_location=device
    ))
    return model

################################################################################   
def train(train_data_loader, model, optimizer, train_loss_list, train_loss_hist, train_itr, DEVICE):
    """
    Performs one epoch of training on the provided model.

    Inputs:
        train_data_loader (DataLoader):    Iterable over training dataset batches.
        model (torch.nn.Module):           The model to train.
        optimizer (torch.optim.Optimizer): Optimizer used for updating model weights.
        train_loss_list (list):            List to store loss values per iteration.
        train_loss_hist (generator):       Generator to track or log loss history.
        train_itr (int):                   Current training iteration count.
        DEVICE (torch.device):             Device (CPU or GPU) to perform computations on.

    Outputs:
        train_loss_list (list): Updated list of training loss values.
    """
    print('Training')
    #global train_itr
    
     # initialize tqdm progress bar
    #prog_bar = Tqdm(train_data_loader, total=len(train_data_loader))
    
    for i, data in enumerate(train_data_loader):
        optimizer.zero_grad()
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)
        train_loss_hist.send(loss_value)
        losses.backward()
        optimizer.step()
        train_itr += 1
    
        # update the loss value beside the progress bar for each iteration
        #prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return train_loss_list

################################################################################   
def validate(valid_data_loader, model, val_loss_list, val_loss_hist, val_itr, DEVICE):
    """
    Performs one epoch of validation on the provided model.

    Inputs:
        valid_data_loader (DataLoader): Iterable over validation dataset batches.
        model (torch.nn.Module):   The model to validate.
        val_loss_list (list):      List to store validation loss values per iteration.
        val_loss_hist (generator): Generator to track or log validation loss history.
        val_itr (int):             Current validation iteration count.
        DEVICE (torch.device):     Device (CPU or GPU) to perform computations on.

    Outputs:
        val_loss_list (list): Updated list of validation loss values.
    """
    print('Validating')
    #global val_itr
    
    # initialize tqdm progress bar
    #prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    
    for i, data in enumerate(valid_data_loader):
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        val_loss_hist.send(loss_value)
        val_itr += 1
        # update the loss value beside the progress bar for each iteration
        #prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return val_loss_list

################################################################################   
# this class keeps track of the training and validation loss values...
# ... and helps to get the average for each epoch as well
class Averager:
    """
    Utility class to compute and maintain the running average of numeric values.

    Methods:
        send(value):      Add a new value to the running total and increment count.
        value (property): Returns the current average of all values received.
        reset():          Resets the total and count to start a new average calculation.
    """
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0
        
    def send(self, value):
        self.current_total += value
        self.iterations += 1
    
    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations
    
    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

################################################################################   
def train_model(model, train_loader, valid_loader, DEVICE, MODEL_NAME, NUM_EPOCHS, OUT_DIR, PLOT_DIR, SAVE_MODEL_EPOCH, SAVE_PLOTS_EPOCH, tqdm_all, train_loss_mpl):
    """
    Train a given PyTorch model with training and validation datasets, 
    periodically saving model checkpoints and loss plots.

    Inputs:
        model (torch.nn.Module):       The model to train.
        train_loader (DataLoader):     DataLoader for training dataset.
        valid_loader (DataLoader):     DataLoader for validation dataset.
        DEVICE (torch.device):         Device on which to run training (CPU or GPU).
        MODEL_NAME (str):              Base name for saving model checkpoints.
        NUM_EPOCHS (int):              Total number of epochs for training.
        OUT_DIR (str):                 Directory to save model checkpoints.
        PLOT_DIR (str):                Directory to save training/validation loss plots.
        SAVE_MODEL_EPOCH (int):        Frequency (in epochs) to save the model.
        SAVE_PLOTS_EPOCH (int):        Frequency (in epochs) to save the loss plots.
        tqdm_all (iterable):           Iterable (e.g., tqdm wrapper) for epoch iteration.
        train_loss_mpl (Panel object): Matplotlib figure holder for live loss plot updates.

    Output:
        list: A list containing two elements:
            - train_loss_list: Loss values for all training iterations.
            - val_loss_list:   Loss values for all validation iterations.
    """
    model = model.to(DEVICE)
    # get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]
    # define the optimizer
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    # initialize the Averager class
    train_loss_hist = Averager()
    val_loss_hist = Averager()
    train_itr = 1
    val_itr = 1
    # train and validation loss lists to store loss values of all...
    # ... iterations till ena and plot graphs for all iterations
    train_loss_list = []
    train_loss_plot_list = []
    val_loss_list = []
    val_loss_plot_list = []
    for epoch, data in enumerate(tqdm_all):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")
        # reset the training and validation loss histories for the current epoch
        train_loss_hist.reset()
        val_loss_hist.reset()
        # create two subplots, one for each, training and validation
        figure_1, train_ax = plt.subplots()
        figure_2, valid_ax = plt.subplots()
        figure, ax = plt.subplots(figsize=(4, 4))  # Make the figure smaller (6x4 inches)
        # start timer and carry out training and validation
        start = time.time()
        train_loss = train(train_loader, model, optimizer, train_loss_list, train_loss_hist, train_itr, DEVICE)
        val_loss = validate(valid_loader, model, val_loss_list, val_loss_hist, val_itr, DEVICE)
        print(f"Epoch #{epoch} train loss: {train_loss_hist.value:.3f}")   
        print(f"Epoch #{epoch} validation loss: {val_loss_hist.value:.3f}")   
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")
        if (epoch+1) % SAVE_MODEL_EPOCH == 0: # save model after every n epochs
            torch.save(model.state_dict(), f"{OUT_DIR}/{MODEL_NAME}{epoch+1}.pth")
            print('SAVING MODEL COMPLETE...\n')
    
        if (epoch+1) % SAVE_PLOTS_EPOCH == 0: # save loss plots after n epochs
            train_ax.plot(train_loss, color='blue')
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            valid_ax.plot(val_loss, color='red')
            valid_ax.set_xlabel('iterations')
            valid_ax.set_ylabel('validation loss')
            # Calculate the average training and validation loss for this epoch
            avg_train_loss = train_loss_hist.value
            avg_val_loss = val_loss_hist.value
            
            # Append the average loss to a list to store epoch-wise loss values
            train_loss_plot_list.append(avg_train_loss)
            val_loss_plot_list.append(avg_val_loss)
             # Validation loss is plotted after every epoch, so scale its x-axis based on epochs
            #validation_x_scale = [(i + 1) * len(train_loader) for i in range(len(val_loss))]  # Scale for validation intervals
            # Generate x-values for validation loss so it scales with the train loss
            # Generate x-axis values representing the epoch numbers
            epochs = range(1, len(train_loss_plot_list) + 1)

            # Plot training and validation loss averages per epoch
            ax.plot(epochs,train_loss_plot_list, label='Average Training Loss', color='blue', marker='o')
            ax.plot(epochs,val_loss_plot_list, label='Average Validation Loss', color='red', marker='o')

            # Set labels and add legend
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Average Loss')
            ax.legend(loc='upper right')

            # Apply tight layout
            plt.tight_layout()
            # Update the Matplotlib object in Panel to show the figure
            train_loss_mpl.object = figure

            #train_loss_mpl.object = figure_1
            figure_1.savefig(f"{PLOT_DIR}/train_loss_{epoch+1}.png")
            figure_2.savefig(f"{PLOT_DIR}/valid_loss_{epoch+1}.png")
            print('SAVING PLOTS COMPLETE...')
    
        if (epoch+1) == NUM_EPOCHS: # save loss plots and model once at the end
            train_ax.plot(train_loss, color='blue')
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            valid_ax.plot(val_loss, color='red')
            valid_ax.set_xlabel('iterations')
            valid_ax.set_ylabel('validation loss')
            figure_1.savefig(f"{PLOT_DIR}/train_loss_{epoch+1}.png")
            figure_2.savefig(f"{PLOT_DIR}/valid_loss_{epoch+1}.png")
            torch.save(model.state_dict(), f"{OUT_DIR}/model{epoch+1}.pth")
    
        plt.close('all')
    return [train_loss_list, val_loss_list]

################################################################################   
def train_model_no_val(model, train_loader, valid_loader, DEVICE, MODEL_NAME, NUM_EPOCHS, OUT_DIR, PLOT_DIR, SAVE_MODEL_EPOCH, SAVE_PLOTS_EPOCH):
    """
    Train a PyTorch model using only the training dataset, without validation.

    Inputs:
        model (torch.nn.Module):   The model to train.
        train_loader (DataLoader): DataLoader for training dataset.
        valid_loader (DataLoader): DataLoader for validation dataset (unused).
        DEVICE (torch.device):     Device for training (CPU or GPU).
        MODEL_NAME (str):          Base filename for saving model checkpoints.
        NUM_EPOCHS (int):          Number of training epochs.
        OUT_DIR (str):             Directory to save model checkpoints.
        PLOT_DIR (str):            Directory to save training loss plots.
        SAVE_MODEL_EPOCH (int):    Frequency (in epochs) to save the model.
        SAVE_PLOTS_EPOCH (int):    Frequency (in epochs) to save training loss plots.

    Output:
        list: A list containing two elements:
            - train_loss_list: Loss values for all training iterations.
            - val_loss_list:   Empty list (validation losses not tracked).
    """
    model = model.to(DEVICE)
    # get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]
    # define the optimizer
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    # initialize the Averager class
    train_loss_hist = Averager()
    val_loss_hist = Averager()
    train_itr = 1
    val_itr = 1
    # train and validation loss lists to store loss values of all...
    # ... iterations till ena and plot graphs for all iterations
    train_loss_list = []
    val_loss_list = []
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")
        # reset the training and validation loss histories for the current epoch
        train_loss_hist.reset()
        #val_loss_hist.reset()
        # create two subplots, one for each, training and validation
        figure_1, train_ax = plt.subplots()
        #figure_2, valid_ax = plt.subplots()
        # start timer and carry out training and validation
        start = time.time()
        train_loss = train(train_loader, model, optimizer, train_loss_list, train_loss_hist, train_itr, DEVICE)
        #val_loss = validate(valid_loader, model, val_loss_list, val_loss_hist, val_itr, DEVICE)
        print(f"Epoch #{epoch} train loss: {train_loss_hist.value:.3f}")   
        #print(f"Epoch #{epoch} validation loss: {val_loss_hist.value:.3f}")   
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")
        if (epoch+1) % SAVE_MODEL_EPOCH == 0: # save model after every n epochs
            torch.save(model.state_dict(), f"{OUT_DIR}/{MODEL_NAME}{epoch+1}.pth")
            print('SAVING MODEL COMPLETE...\n')
    
        if (epoch+1) % SAVE_PLOTS_EPOCH == 0: # save loss plots after n epochs
            train_ax.plot(train_loss, color='blue')
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            #valid_ax.plot(val_loss, color='red')
            #valid_ax.set_xlabel('iterations')
            #valid_ax.set_ylabel('validation loss')
            figure_1.savefig(f"{PLOT_DIR}/train_loss_{epoch+1}.png")
            #figure_2.savefig(f"{PLOT_DIR}/valid_loss_{epoch+1}.png")
            print('SAVING PLOTS COMPLETE...')
    
        if (epoch+1) == NUM_EPOCHS: # save loss plots and model once at the end
            train_ax.plot(train_loss, color='blue')
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            #valid_ax.plot(val_loss, color='red')
            #valid_ax.set_xlabel('iterations')
            #valid_ax.set_ylabel('validation loss')
            figure_1.savefig(f"{PLOT_DIR}/train_loss_{epoch+1}.png")
            #figure_2.savefig(f"{PLOT_DIR}/valid_loss_{epoch+1}.png")
            torch.save(model.state_dict(), f"{OUT_DIR}/model{epoch+1}.pth")
    
        plt.close('all')
    return [train_loss_list, val_loss_list]

