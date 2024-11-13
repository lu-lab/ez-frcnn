import sys
from pathlib import Path
# Add the parent directory to the Python path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from library import *
import panel as pn
import ipywidgets
from panel.widgets import Tqdm
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import os

pn.extension('ipywidgets')
tqdm = Tqdm()
fig = plt.figure(figsize=(4, 4))

inf_fig = pn.pane.Matplotlib(fig)
inf_fig.margin = (-30, 0, 0, -15) 

# Function to automatically retrieve class names from annotations
def get_class_names_from_annotations(train_dir, test_dir):
    class_names = set()
    xml_files = glob.glob(os.path.join(train_dir, "*.xml"))
    xml_files.extend(glob.glob(os.path.join(test_dir, "*.xml")))

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for obj in root.findall('object'):
            class_name = obj.find('label').text
            class_names.add(class_name)

    return sorted(list(class_names))

# Confidence threshold input with a clear description
threshold = pn.widgets.FloatInput(
    name='Confidence Threshold', 
    value=0.9, 
    step=1e-2, 
    start=0, 
    end=1,
    description="Set the confidence level above which detections are considered valid. For example, 0.9 means only detections with 90% confidence or above are kept."
)

# Model name input with a better description
model_name = ipywidgets.Text(
    value='model20.pth',
    placeholder='Model File Name',
    description='Model File Name:',
    disabled=False   
)

# Button for inference with a better description
inference_button = pn.widgets.Button(
    name='Run Inference on Test Data',
    button_type='primary',
    margin=(20, 0, 20, 0)
)

# Inference function when button is clicked
def handle_inf_images(clicks):
    if clicks > 0:
        # Automatically get class names from the train and test annotations
        class_names = get_class_names_from_annotations('./images/train', './images/test')
        print(f"Detected class names: {class_names}")

        # Load the model and run inference
        model = load_model(model_name.value, MODEL_DIR, len(class_names) + 1)  # Add 1 for background
        [boxFileName, classFileName, scoreFileName] = ['boxes', 'classes', 'scores']
        # List to store results for CSV
        results = []

        # Run inference on the images and return bounding boxes, classes, and scores
        [bboxes, classes, scores] = inference_images(
            './test_data/test_images/', model, './outputs/', threshold.value, ['background'] + class_names, tqdm,
            inf_fig
        )
        

        return

# Bind the inference button to the handler
te = pn.Column(pn.bind(handle_inf_images, inference_button.param.clicks))
inf_button = pn.Row(pn.Spacer(width=60), inference_button)
# Define the side panel with updated descriptions
side = pn.Column(
    pn.pane.Markdown("### Inference Parameters:"),
    pn.pane.Markdown("1. **Confidence Threshold:** Determines the minimum confidence score for a detected object to be considered valid."),
    threshold,
    pn.pane.Markdown("2. **Model File Name:** Specify the name of the trained model file (e.g., 'model20.pth'). The model should be placed in the model directory."),
    model_name,
    pn.pane.Markdown("### Instructions:"),
    pn.pane.Markdown("1. Place the images to be tested in the folder: `./test_data/test_images/`"),
    pn.pane.Markdown("2. The output will be saved in the folder: `./outputs/`."),
    pn.pane.Markdown("3. The results will be saved as a CSV file in the following format:\n"
                     "    - **Image Name:** The name of the image file.\n"
                     "    - **Bounding Boxes:** Coordinates of the detected bounding boxes.\n"
                     "    - **Classes:** The predicted classes of the objects.\n"
                     "    - **Scores:** Confidence scores for the detected objects."),
    inf_button  # Add the button here so it's displayed
)
out = pn.Column(tqdm,te,inf_fig)
# Create the layout and serve
pn.template.MaterialTemplate(
    site="EZ-FRCNN",
    title="Inferencing",
    sidebar=[side],
    main=[out],  # Panel to display tqdm progress, results, and figure
).servable()
