from jupyter_bbox_widget import BBoxWidget
import ipywidgets as widgets
from ipywidgets import Layout
import os
import json
import glob
import xmltodict
import shutil
import random
from IPython.display import display, Javascript
import cv2
import base64
import pandas as pd

def encode_image(filepath):
    with open(filepath, 'rb') as f:
        image_bytes = f.read()
    encoded = str(base64.b64encode(image_bytes), 'utf-8')
    return "data:image/png;base64,"+encoded

def init_annotations(classes):
    global files, annotations, current_index  # Ensure these are accessible globally
    path = './annotations/'
    current_index = 0  # Initialize the index variable
    annotations = {}
    image_extensions = ['jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp']
    all_extensions = image_extensions + [ext.upper() for ext in image_extensions]  # Add uppercase versions
    files = glob.glob(f"{path}/*.png")
    for extension in all_extensions:
        files.extend(glob.glob(f"{path}/*.{extension}"))
    files = [os.path.basename(image_path) for image_path in files]  # More portable than split('/')
    if not files:
        print("No images found in the specified directory.")
        return
    im = encode_image(os.path.join(path, files[0]))
    w_bbox = BBoxWidget(
        image=im,
        classes=classes,
        hide_buttons=True
    )

    # Initialize the label widget here to update it in the functions
    label = widgets.Label(value=files[0])

    # Define the function to update the image and label
    def update_image_and_annotations():
        nonlocal label  # Indicates that we're using the `label` defined outside this function
        global current_index
        if current_index < len(files):
            image_file = encode_image(os.path.join(path, files[current_index]))
            w_bbox.image = image_file  # Update the image in the widget
            w_progress.value = current_index  # Update the progress bar
            w_progress.description = f"{current_index + 1} / {len(files)}"  # Update the progress bar name with index/total
            label.value = files[current_index]  # Update the label
        else:
            print("Reached the end of the file list.")

    def go_back(_):
        global current_index
        if current_index > 0:
            current_index -= 1
            w_bbox.bboxes = []
            update_image_and_annotations()

    def skip(_):
        global current_index
        if current_index < len(files) - 1:
            current_index += 1
            w_bbox.bboxes = []
            update_image_and_annotations()

            
    def submit(_):
        global annotations, current_index
        image_file = files[w_progress.value]
        path = './annotations/'
        #print(image_file)
        
        image = cv2.imread(os.path.join(path, image_file))
        height, width, _ = image.shape
        # save annotations for current image
        annotations[image_file] = w_bbox.bboxes
        if annotations[image_file]:
            annotations_for_image = {'annotation': {'object': annotations[image_file]}}
        else:
            # No annotations, create a 'background' label with full image bounding box
            background_annotation = {
                
                    'x': 0,
                    'y': 0,
                    'width': width,
                    'height': height,
                    'label':'background'
                
            }
            annotations_for_image = {'annotation': {'object': background_annotation}}

        xml_string = xmltodict.unparse(annotations_for_image, pretty=True)
        with open(path + image_file[:-4] + '.xml', 'w') as f:
            #json.dump(annotations[image_file], f, indent=4)
            f.write(xml_string)
        # move on to the next file
        skip(_)
    
    def on_reset_clicked(b):
        w_bbox.bboxes = []  # Clears the drawn boxes in your widget

    # Connect buttons to their callbacks
    go_back_button = widgets.Button(description="Go Back", layout=Layout(width='auto'))
    go_back_button.on_click(go_back)
    
    skip_button = widgets.Button(description="Skip", layout=Layout(width='auto'))
    skip_button.on_click(skip)
    
    reset_button = widgets.Button(description="Reset", layout=Layout(width='auto'))
    reset_button.on_click(on_reset_clicked)
    
    submit_button = widgets.Button(description="Submit", layout=Layout(width='auto'))
    submit_button.on_click(submit)

    # Progress bar to show how far we got
    # Progress bar to show how far we got, moved slightly to the left using margin
    w_progress = widgets.IntProgress(
        value=0, 
        max=len(files)-1, 
        description=f'1 / {len(files)}',
        layout=widgets.Layout(width='287px', margin='0 0 0 -40px')  # Adjust width and left margin
    )

    
    # Combine widgets into containers
    navigation_buttons = widgets.HBox([go_back_button, skip_button, submit_button, reset_button], layout=Layout(margin='0 0 0 3px'))
    w_progressLabeled = widgets.HBox([w_progress, label])
    
    w_container = widgets.VBox([
        w_progressLabeled,
        navigation_buttons,
        w_bbox,
        
    ])
   
    return w_container


def split_images_and_xml(source_folder, train_folder='./images/train/', test_folder='./images/test/', test_ratio=0.1):
    """
    Splits images and their corresponding XML files into training and testing folders randomly.
    Outputs individual CSV files for each image's annotations, along with the XML files.
    
    Parameters:
    - source_folder: Folder containing the images and XML files.
    - train_folder: Destination folder for the training split.
    - test_folder: Destination folder for the testing split.
    - test_ratio: Fraction of data to be used as test set. Default is 0.1 for 10%.
    """
    # Ensure the train and test folders exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    image_extensions = ['jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp']
    all_extensions = image_extensions + [ext.upper() for ext in image_extensions]  # Add uppercase versions
    # Find all image files in the source folder. .
    image_files = glob.glob(os.path.join(source_folder, "*.png"))
    path = './annotations'
    for extension in all_extensions:
        image_files.extend(glob.glob(f"{path}/*.{extension}"))

    # Shuffle the image files to ensure random selection
    random.shuffle(image_files)

    # Calculate the number of files to allocate to the test set
    num_test_files = int(len(image_files) * test_ratio)

    # Split the files into training and testing sets
    test_files = image_files[:num_test_files]
    train_files = image_files[num_test_files:]

    # Function to convert XML annotations to CSV format
    def xml_to_csv(xml_file, output_csv_file):
        with open(xml_file) as xml_fd:
            annotation_data = xmltodict.parse(xml_fd.read())
            objects = annotation_data.get('annotation', {}).get('object', [])
            
            # Handle case if only one object exists (not a list)
            if not isinstance(objects, list):
                objects = [objects]

            csv_rows = []
            for obj in objects:
                if obj is not None:
                    class_name = obj['label']
                    x = obj['x']
                    y = obj['y']
                    width = obj['width']
                    height = obj['height']
                    csv_rows.append([class_name, x, y, width, height])

            # Convert rows to a pandas DataFrame and save to CSV
            df = pd.DataFrame(csv_rows, columns=['class', 'xmin', 'ymin', 'xmax', 'ymax'])
            df.to_csv(output_csv_file, index=False)

    # Copy the files to their respective folders and generate CSV annotations
    for file_path in test_files:
        shutil.copy(file_path, test_folder)
        xml_path = os.path.splitext(file_path)[0] + '.xml'
        if os.path.exists(xml_path):
            shutil.copy(xml_path, test_folder)
            csv_path = os.path.splitext(file_path)[0] + '.csv'
            xml_to_csv(xml_path, os.path.join(test_folder, os.path.basename(csv_path)))
    
    for file_path in train_files:
        shutil.copy(file_path, train_folder)
        xml_path = os.path.splitext(file_path)[0] + '.xml'
        if os.path.exists(xml_path):
            shutil.copy(xml_path, train_folder)
            csv_path = os.path.splitext(file_path)[0] + '.csv'
            xml_to_csv(xml_path, os.path.join(source_folder, os.path.basename(csv_path)))

