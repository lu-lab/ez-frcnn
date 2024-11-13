from torch.utils.data import Subset
import numpy as np
import torch
from torchvision.ops import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt
from .utils import *
from .training import *

def get_subsampled_dataset(full_dataset, num_samples):
    """ Randomly subsample the dataset to the specified number of samples. """
    indices = np.random.permutation(len(full_dataset))[:num_samples]
    return Subset(full_dataset, indices)

def get_mAP(dataloader, model, device):
    metric = MeanAveragePrecision()
    model.eval()
    model.to(device)
    # Iterate over the DataLoader
    for images, targets in dataloader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            outputs = model(images)

        # Prepare predictions and targets for torchmetrics
        preds = []
        for output in outputs:
            preds.append({
                "boxes": output['boxes'].cpu(),
                "scores": output['scores'].cpu(),
                "labels": output['labels'].cpu(),
            })

        tars = []
        for target in targets:
            tars.append({
                "boxes": target['boxes'].cpu(),
                "labels": target['labels'].cpu(),
            })

        # Update the metric with predictions and targets
        metric.update(preds, tars)

    # Compute the final mAP score
    mAP = metric.compute()
    return mAP

    

def run_experiment(full_train_dataset, valid_dataset, num_classes, BATCH_SIZE, NUM_EXPERIMENTS=5, EPOCHS_PER_EXPERIMENT=100, TRIALS_PER_EXPERIMENT=3):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metric = MeanAveragePrecision()
    # Directory setup for models and plots
    model_dir = 'models/'
    plot_dir = 'plots/'

    # Calculate number of samples for each experiment
    total_samples = len(full_train_dataset)
    split_sizes = np.linspace(0, total_samples, NUM_EXPERIMENTS + 1, dtype=int)[1:]

    results = []

    # Create plot
    plt.figure()
    mean_mAPs = []
    std_mAPs = []

    for num_samples in split_sizes:
        mAPs = []
        for trial in range(TRIALS_PER_EXPERIMENT):
            print(f"\nRunning training with {num_samples} samples, trial {trial + 1}...")

            # Subsample the training dataset
            train_subset = get_subsampled_dataset(full_train_dataset, num_samples)
            train_loader, valid_loader = get_loaders(train_subset, valid_dataset, BATCH_SIZE, collate_fn)

            # Initialize a fresh instance of the model
            model = create_model(num_classes).to(device)

            # Train the model
            train_model_no_val(model, train_loader, valid_loader, device, 'experiment_model', EPOCHS_PER_EXPERIMENT, model_dir, plot_dir, 10, 10)

            # Evaluate the model
            val_mAP = get_mAP(valid_loader, model, device)
            val_mAP = float(val_mAP['map_50'])
            mAPs.append(val_mAP)
            print(f"Trial {trial + 1}: Validation mAP = {val_mAP:.3f}")

        # Compute statistics
        mean_mAP = np.mean(mAPs)
        std_mAP = np.std(mAPs)
        mean_mAPs.append(mean_mAP)
        std_mAPs.append(std_mAP)
        results.append((num_samples, mean_mAP, std_mAP))
        print(f"Finished {num_samples} samples: Mean Validation mAP = {mean_mAP:.3f}, Std Dev = {std_mAP:.3f}")

    # Plotting results
    plt.errorbar(split_sizes, mean_mAPs, yerr=std_mAPs, fmt='-o', capsize=5)
    plt.title('Mean and Standard Deviation of Validation mAP')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Validation mAP')
    plt.grid(True)
    plt.savefig(f"{plot_dir}/mAP_results.png")
    plt.show()

    return results