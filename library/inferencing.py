################################################################################
# inferencing.py
# Written by Jacob Wheelock & Erin Shappell for Lu Lab
# 
# This module provides helper functions for evaluating trained object detection models,
# saving results, and running inference on video data.
#
################################################################################
# Imports
import numpy as np
import torch
import cv2
import glob as glob
from .training import create_model
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
import os
from .utils import *
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

################################################################################   
def load_model(model_name, MODEL_DIR, NUM_CLASSES):
    """
    Loads a trained model from disk and prepares it for evaluation.

    Inputs:
        model_name (str):  Filename of the saved model weights.
        MODEL_DIR (str):   Directory path where the model files are stored.
        NUM_CLASSES (int): Number of output classes for the model.

    Output:
        nn.Module: The loaded PyTorch model set to evaluation mode on the appropriate device.
    """
    # set the computation device
    modelPath = './models/' + model_name
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # load the model and the trained weights
    model = create_model(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(
        modelPath, map_location=device
    ))
    model.eval()
    return model

################################################################################   
def saveResultsToCSV(csvFileName, results, OUT_DIR):
    """
    Saves detection results to a CSV file with specified columns.

    Inputs:
        csvFileName (str):      Name of the CSV file (without extension) to save results.
        results (list of dict): List of detection result dictionaries containing keys
                                'image_name', 'boxes', 'classes', and 'scores'.
        OUT_DIR (str):          Directory path where the CSV file will be saved.

    Output:
        None: Writes the results to a CSV file at the specified location.
    """
    csv_path = os.path.join(OUT_DIR, f"{csvFileName}.csv")
    
    # Open CSV file and write the data
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(['Image Name', 'Bounding Boxes', 'Classes', 'Scores'])  # CSV Header
        
        for result in results:
            writer.writerow([result['image_name'], result['boxes'], result['classes'], result['scores']])

################################################################################               
def inference_video(DIR_TEST, OUT_DIR, vidName, model, detection_threshold, CLASSES, save_detections=False):
    """
    Runs object detection on a video, annotates detected objects frame-by-frame, 
    optionally saves detected regions, and writes the annotated video to disk.

    Inputs:
        DIR_TEST (str):                   Path to the input video file for inference.
        OUT_DIR (str):                    Directory where output video and detected regions (optional) will be saved.
        vidName (str):                    Filename for the output annotated video.
        model (torch.nn.Module):          Trained object detection model.
        detection_threshold (float):      Confidence threshold for filtering detections.
        CLASSES (list):                   List of class names corresponding to model outputs.
        save_detections (bool, optional): If True, saves detected bounding box regions as separate images. Default is False.

    Outputs:
        list: A list containing three elements for all frames:
            - bboxes (list): Detected bounding boxes per frame.
            - classes (list): Detected class labels per frame.
            - sscores (list): Detection scores per frame.
    """
    vid = cv2.VideoCapture(DIR_TEST)
    property_id = int(cv2.CAP_PROP_FRAME_COUNT) 
    NUM_FRAMES = int(cv2.VideoCapture.get(vid, property_id))
    idx = 1
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter((OUT_DIR + '/' + vidName),cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
    classes = [None] * NUM_FRAMES
    bboxes = [None] * NUM_FRAMES
    sscores = [None] * NUM_FRAMES
    
    while vid.isOpened():
        ret, image = vid.read()
        
        orig_image = image.copy()
        # BGR to RGB
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # make the pixel range between 0 and 1
        image /= 255.0
        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(float)
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float).cuda()
        # add batch dimension
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            outputs = model(image)
        
        # load all detection to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        # carry further only if there are detected boxes
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            sscores[idx] = scores
            
            # filter out boxes according to `detection_threshold`
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            bboxes[idx] = boxes
            draw_boxes = bboxes[idx].copy() 
             
            # get all the predicited class names
            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
            pred_classes = np.array(pred_classes)
            pred_classes = pred_classes[scores >= detection_threshold]
            classes[idx] = pred_classes
            
            if (save_detections):
                for j, box in enumerate(draw_boxes):
                    # Extract and save each detected region
                    detected_region = orig_image[box[1]:box[3], box[0]:box[2]]
                    region_save_path = f"{OUT_DIR}/frame_{idx:04d}_box_{j:02d}.png"
                    cv2.imwrite(region_save_path, detected_region)
            # draw the bounding boxes and write the class name on top of it
            for j, box in enumerate(draw_boxes):
                cv2.rectangle(orig_image,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            (0, 0, 255), 2)
                cv2.putText(orig_image, str(pred_classes[j]), 
                            (int(box[0]), int(box[1]-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 
                            2, lineType=cv2.LINE_AA)
            out.write(orig_image)
        idx += 1
        print(f"Image {idx+1} done...")
        print('-'*50)
        if idx == NUM_FRAMES:
            vid.release()
            out.release()
    print('TEST PREDICTIONS COMPLETE') 
    return [bboxes, classes, sscores]

################################################################################   
def inference_images(DIR_TEST, model, OUT_DIR, detection_threshold, CLASSES, tqdmBar, inf_fig):
    """
    Performs object detection on all images in a specified directory, annotates and saves the results, 
    and records detection details for further analysis.

    Inputs:
        DIR_TEST (str):              Path to the directory containing input images.
        model (torch.nn.Module):     Trained object detection model.
        OUT_DIR (str):               Directory where annotated images and results CSV will be saved.
        detection_threshold (float): Confidence threshold for filtering detections.
        CLASSES (list):              List of class names corresponding to model output labels.
        tqdmBar (callable):          Progress bar function for iterating over images.
        inf_fig (object):            Visualization object used to display annotated images.

    Outputs:
        list: A list of dictionaries, each containing:
            - 'image_name' (str): Filename of the image.
            - 'boxes' (list):     Detected bounding boxes as lists of coordinates.
            - 'classes' (list):   Predicted class labels.
            - 'scores' (list):    Confidence scores for detections.
    """
    imagePath = glob.glob(f"{DIR_TEST}/*.png")
    image_extensions = ['jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp', 'tif']
    all_extensions = image_extensions + [ext.upper() for ext in image_extensions]  # Add uppercase versions
    for extension in all_extensions:
            imagePath.extend(glob.glob(f"{DIR_TEST}/*.{extension}"))
    all_images = [image_path.split('/')[-1] for image_path in imagePath]
    all_images = sorted(all_images)
    num_images = len(all_images)
    classes = [None] * num_images
    bboxes = [None] * num_images
    sscores = [None] * num_images
    # List to store results for CSV
    results = []
    for idx in tqdmBar(range(0,num_images)):
        el = all_images[idx]
        orig_image = cv2.imread(DIR_TEST + '/' + el)
        # BGR to RGB
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # make the pixel range between 0 and 1
        image /= 255.0
        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(float)
        # convert to tensor
        if torch.cuda.is_available():
            image = torch.tensor(image, dtype=torch.float).cuda()
        else:
            image = torch.tensor(image, dtype=torch.float)
        # add batch dimension
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            outputs = model(image)
        
        # load all detection to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        # carry further only if there are detected boxes
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            sscores[idx] = scores[scores >= detection_threshold]
            
            # filter out boxes according to `detection_threshold`
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            bboxes[idx] = boxes
            draw_boxes = bboxes[idx].copy() 
             
            # get all the predicited class names
            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
            pred_classes = np.array(pred_classes)
            pred_classes = pred_classes[scores >= detection_threshold]
            classes[idx] = pred_classes
            # Store results for this image in the list
            results.append({
                'image_name': el,
                'boxes': boxes.tolist(),
                'classes': pred_classes.tolist(),
                'scores': sscores[idx].tolist()
            })
            # draw the bounding boxes and write the class name on top of it
            fig, ax = plt.subplots(1, figsize=(4,4))
            ax.axis('off')
            orig_image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            plt.tight_layout()
            ax.imshow(orig_image_rgb)
            
            inf_fig.object = fig
            for j, box in enumerate(draw_boxes):
                cv2.rectangle(orig_image_rgb,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            (255, 0, 0), 5)
                cv2.putText(orig_image_rgb, str(pred_classes[j]), 
                            (int(box[0]), int(box[1]-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 
                            2, lineType=cv2.LINE_AA)
            writeOut = cv2.cvtColor(orig_image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(OUT_DIR + '/' + el, writeOut) #The 'el' filepath is broken right now (TODO: FIX) 
            ax.axis('off')  # Remove the axis for cleaner visualization
            plt.tight_layout()
            ax.imshow(orig_image_rgb)
            # Update the inf_fig pane with the new figure
            inf_fig.object = fig
            plt.close()
        #print(f"Image {idx+1} done...")
        #print('-'*50)
    
    saveResultsToCSV('inference_results', results, OUT_DIR)
    print('TEST PREDICTIONS COMPLETE') 
    return results

################################################################################   
def load_and_preprocess_image(file_path, target_size=(800, 800)):
    """
    Loads an image from disk, resizes it to a target size, converts it to RGB, normalizes pixel values,
    and transforms it into a PyTorch tensor suitable for model input.

    Inputs:
        file_path (str):     Path to the input image file.
        target_size (tuple): Desired output image size as (width, height). Default is (800, 800).

    Outputs:
        tuple: A tuple containing:
            - image_tensor (torch.Tensor): Preprocessed image tensor of shape (3, target_height, target_width).
            - filename (str):              The basename of the input image file.
            - original_size (tuple):       Original image dimensions as (width, height).
    """
    orig_image = cv2.imread(file_path)
    orig_height, orig_width = orig_image.shape[:2]
    resized_image = cv2.resize(orig_image, target_size)  # Resize to fixed size
    image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image_tensor = torch.tensor(np.transpose(image_rgb, (2, 0, 1)), dtype=torch.float)
    return image_tensor, os.path.basename(file_path), (orig_width, orig_height)

################################################################################   
def scale_boxes_to_original(boxes, original_size, resized_size=(800, 800)):
    """
    Scales bounding box coordinates from a resized image back to the original image dimensions.

    Inputs:
        boxes (array-like):    Array of bounding boxes with coordinates [x_min, y_min, x_max, y_max] 
                               relative to the resized image.
        original_size (tuple): Original image size as (width, height).
        resized_size (tuple):  Resized image size as (width, height). Default is (800, 800).

    Outputs:
        numpy.ndarray: Array of bounding boxes scaled to the original image size.
    """
    orig_width, orig_height = original_size
    resized_width, resized_height = resized_size
    x_scale = orig_width / resized_width
    y_scale = orig_height / resized_height
    scaled_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        scaled_boxes.append([
            x_min * x_scale, y_min * y_scale,
            x_max * x_scale, y_max * y_scale
        ])
    return np.array(scaled_boxes)

################################################################################   
def inference_images_fast(DIR_TEST, model, OUT_DIR, detection_threshold, CLASSES, tqdmBar, batch_size=4):
    """
    Performs batch inference on images in a directory using the provided model, with optional GPU acceleration.

    Inputs:
        DIR_TEST (str):              Directory path containing images for inference.
        model (torch.nn.Module):     Trained object detection model.
        OUT_DIR (str):               Directory path to save inference results.
        detection_threshold (float): Minimum confidence score to consider a detection valid.
        CLASSES (list):              List of class names corresponding to model labels.
        tqdmBar (iterable):          Progress bar iterator for displaying progress.
        batch_size (int, optional):  Number of images to process per batch. Default is 4.

    Outputs:
        list of dict: Each dict contains image filename, bounding boxes (scaled to original image size), 
                      predicted classes, and detection scores for that image.
    """
    # Collect all image paths
    image_extensions = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp']
    all_image_paths = []
    for ext in image_extensions + [ext.upper() for ext in image_extensions]:
        all_image_paths.extend(glob.glob(f"{DIR_TEST}/*.{ext}"))
    all_image_paths = sorted(all_image_paths)

    # Prepare results list for annotations
    results = []
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # Process images in batches
    with ThreadPoolExecutor() as executor:
        for start_idx in tqdmBar(range(0, len(all_image_paths), batch_size), desc="Inference Progress"):
            # Load images in parallel
            batch_paths = all_image_paths[start_idx:start_idx + batch_size]
            batch_data = list(executor.map(load_and_preprocess_image, batch_paths))
            
            # Separate image tensors and filenames
            images, filenames, original_sizes = zip(*batch_data)
            images = torch.stack(images).to(device)
            
            # Run inference
            with torch.no_grad():
                outputs = model(images)
            
            # Process each image output
            for i, output in enumerate(outputs):
                scores = output['scores'].cpu().numpy()
                boxes = output['boxes'][scores >= detection_threshold].cpu().numpy()
                labels = output['labels'][scores >= detection_threshold].cpu().numpy()
                
                # Scale boxes back to original image size
                orig_size = original_sizes[i]
                scaled_boxes = scale_boxes_to_original(boxes, orig_size)
                
                # Store annotation results
                pred_classes = [CLASSES[label] for label in labels]
                result = {
                    'image_name': filenames[i],
                    'boxes': scaled_boxes.tolist(),
                    'classes': pred_classes,
                    'scores': scores[scores >= detection_threshold].tolist()
                }
                results.append(result)

    # Save results to JSON or CSV
    saveResultsToCSV('inference_results', results, OUT_DIR)
    print('TEST PREDICTIONS COMPLETE')
    
    return results

################################################################################   
def inference_images_figs(DIR_TEST, model, OUT_DIR, detection_threshold, CLASSES):
    """
    Performs inference on images in a directory using the given model, annotates detected objects with bounding boxes 
    and class labels, and overlays enlarged views of detected regions on the original images. Saves annotated images with 
    bounding boxes and enlarged detected regions overlaid to OUT_DIR.

    Inputs:
        DIR_TEST (str):              Directory path containing input images.
        model (torch.nn.Module):     Trained object detection model.
        OUT_DIR (str):               Directory path to save annotated output images.
        detection_threshold (float): Minimum confidence score to consider a detection valid.
        CLASSES (list):              List of class names corresponding to model output labels.

    Outputs:
        list: A list containing three elements:
            - bboxes (list):  Detected bounding boxes per image.
            - classes (list): Predicted class labels per image.
            - sscores (list): Detection scores per image.

    """
    imagePath = glob.glob(f"{DIR_TEST}/*.png")
    image_extensions = ['jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp', 'tif']
    all_extensions = image_extensions + [ext.upper() for ext in image_extensions]  # Add uppercase versions
    for extension in all_extensions:
        imagePath.extend(glob.glob(f"{DIR_TEST}/*.{extension}"))

    all_images = [image_path.split('/')[-1] for image_path in imagePath]
    all_images = sorted(all_images)
    num_images = len(all_images)
    classes = [None] * num_images
    bboxes = [None] * num_images
    sscores = [None] * num_images
    
    for idx, el in enumerate(all_images):
        orig_image = cv2.imread(DIR_TEST + '/' + el)
        # BGR to RGB
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # Normalize the pixel values (between 0 and 1)
        image /= 255.0
        # Rearrange color channels
        image = np.transpose(image, (2, 0, 1)).astype(float)
        # Convert to tensor
        image_tensor = torch.tensor(image, dtype=torch.float).cuda() if torch.cuda.is_available() else torch.tensor(image, dtype=torch.float)
        # Add batch dimension
        image_tensor = torch.unsqueeze(image_tensor, 0)
        
        with torch.no_grad():
            outputs = model(image_tensor)
        
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            sscores[idx] = scores[scores >= detection_threshold]
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            bboxes[idx] = boxes
            draw_boxes = boxes.copy() 
            
            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
            pred_classes = np.array(pred_classes)
            pred_classes = pred_classes[scores >= detection_threshold]
            classes[idx] = pred_classes
            
            for j, box in enumerate(draw_boxes):
                x1, y1, x2, y2 = box
                cv2.rectangle(orig_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(orig_image, str(pred_classes[j]), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Extract and enlarge the detected region
                detected_img = orig_image[y1:y2, x1:x2]
                factor = 2  # Change factor to desired zoom
                enlarged_img = cv2.resize(detected_img, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)
                
                # Calculate where to place the enlarged image on the original
                eh, ew, _ = enlarged_img.shape
                ex, ey = 10, 10  # Starting coordinates for the enlarged image (top left)
                
                # Ensure the enlarged image does not go out of the bounds of the original image
                if ey + eh > orig_image.shape[0]:
                    ey = orig_image.shape[0] - eh
                if ex + ew > orig_image.shape[1]:
                    ex = orig_image.shape[1] - ew
                
                # Overlay the enlarged image on the original image
                orig_image[ey:ey+eh, ex:ex+ew] = enlarged_img
                
                # Draw lines connecting the small and enlarged boxes
                cv2.line(orig_image, (x1, y1), (ex, ey), (255, 0, 0), 2)
                cv2.line(orig_image, (x2, y2), (ex + ew, ey + eh), (255, 0, 0), 2)

            cv2.imwrite(OUT_DIR + '/' + el, orig_image)  # Save the modified image

        print(f"Image {idx+1} done...")
        print('-'*50)

    print('TEST PREDICTIONS COMPLETE') 
    return [bboxes, classes, sscores]
