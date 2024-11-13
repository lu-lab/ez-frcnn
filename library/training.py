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

def create_model(num_classes):
    
    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='COCO_V1')
    
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model

def load_model_train(model_name, MODEL_DIR, NUM_CLASSES):
    # set the computation device
    modelPath = './models/' + model_name
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # load the model and the trained weights
    model = create_model(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(
        modelPath, map_location=device
    ))
    return model

def train(train_data_loader, model, optimizer, train_loss_list, train_loss_hist, train_itr, DEVICE):
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

# function for running validation iterations
def validate(valid_data_loader, model, val_loss_list, val_loss_hist, val_itr, DEVICE):
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

# this class keeps track of the training and validation loss values...
# ... and helps to get the average for each epoch as well
class Averager:
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


def train_model(model, train_loader, valid_loader, DEVICE, MODEL_NAME, NUM_EPOCHS, OUT_DIR, PLOT_DIR, SAVE_MODEL_EPOCH, SAVE_PLOTS_EPOCH, tqdm_all, train_loss_mpl):
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

def train_model_no_val(model, train_loader, valid_loader, DEVICE, MODEL_NAME, NUM_EPOCHS, OUT_DIR, PLOT_DIR, SAVE_MODEL_EPOCH, SAVE_PLOTS_EPOCH):
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

