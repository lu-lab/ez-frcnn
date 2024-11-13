import sys
from pathlib import Path
# Add the parent directory to the Python path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import panel as pn
import ipywidgets
pn.extension('mathjax')
pn.extension('ipywidgets')
from library import *
from panel.widgets import Tqdm
import xml.etree.ElementTree as ET
import os
import glob
pn.extension()
# Initialize widgets for progress and loss plot
tqdm = Tqdm(visible=False, width=600)
fig = plt.figure(figsize=(4, 4))
train_loss_mpl = pn.pane.Matplotlib(fig)
train_loss_mpl.margin = (0, 0, 0, -30) 

# Add the custom CSS to Panel
pn.config.raw_css.append("""
    .center-text {
        text-align: left;
    }
""")

pn.config.raw_css.append("""
    /* Increase the scrollbar size */
    ::-webkit-scrollbar {
        height: 24px;  /* Height for horizontal scrollbar */
        width: 12px;  /* Width for vertical scrollbar */
        background: #555
        background-color: #555
    }

    /* Custom styling for the scrollbar track and thumb */
    ::-webkit-scrollbar-track {
        background-color: #EBEBFF !important;
        
    }

    ::-webkit-scrollbar-thumb {
        background: #555;
        background-color: #555 !important;
        border-radius: 10px;
        background-color: #555
    }

    ::-webkit-scrollbar-thumb:hover {
        background-color: #555;
    }

    /* Force horizontal scrollbar at the bottom of the viewport */
    body, .bk-root {
        overflow-x: scroll;
        overflow-y: hidden;
        font-size: 16px !important;
        
    }
    
    /* Ensure the scrollbar is visible at the bottom of the viewport */
    html, body {
        height: 100%;
        width: 100%;
        display: flex;
        flex-direction: column;
        font-size: 16px !important;
        
    }
    

    .pn-main {
        height: 100%;
        flex-grow: 1;
        overflow-y: hidden;
        display: flex;
        overflow-x: scroll;  /* Horizontal scrollbar always visible */
        
    }
    
""")
# Updated descriptions and sections with numbering
# Detailed descriptions for training parameters
text1 = pn.pane.Markdown("""
## 1. Set Training Parameters and Model Name
""", width=330, css_classes=['center-text'],)

# Explanation of each training parameter
batch_size_description = pn.pane.Markdown("""
This controls the number of samples that are processed before the model updates its parameters. A larger batch size can help improve training speed but may require more memory. Typical values range from 4 to 64 depending on your GPU/CPU capacity.
""", width=330)

resize_to_description = pn.pane.Markdown("""
Resize the input images to a smaller resolution before feeding them into the model. This reduces computational cost and memory usage. Typically, a value between 256 and 512 works well for most images.
""", width=330)

num_epochs_description = pn.pane.Markdown(""" 
The total number of times the entire dataset will be passed through the model. A higher number of epochs may improve accuracy but can also lead to overfitting if set too high. A typical range is 20 to 100 epochs depending on the dataset.
""", width=330)

save_plots_epoch_description = pn.pane.Markdown(""" 
This parameter controls how often the loss plots will be saved to disk. For example, if set to 1, the loss plot will be saved after every epoch.
""", width=330)

save_model_epoch_description = pn.pane.Markdown("""
Set the frequency of saving model checkpoints during training. This ensures you have multiple model versions saved at different epochs, in case of interruptions or to analyze performance at different stages.
""", width=330)

new_model_name_description = pn.pane.Markdown(""" 
Provide a custom name for the model. This name will be used when saving model checkpoints and loss plots, allowing you to easily identify different training runs.
""", width=330)

text2 = pn.pane.Markdown("""
## 2. Load and Verify Dataset
Load the dataset and visualize a random sample to ensure the data and annotations are set up correctly before training.
""", width=330, css_classes=['center-text'])

text3 = pn.pane.Markdown("""
## 3. Train Model and Visualize Loss
Train the model and monitor the progress with real-time loss visualization. Model weights and loss plots will be saved at intervals specified in the parameters.
""", width=330, css_classes=['center-text'])

# Initialize parameters for training
BATCH_SIZE = pn.widgets.IntInput(name='Batch Size', value=4, step=1, start=1, end=16)
RESIZE_TO = pn.widgets.IntInput(name='Downsampling Size', value=512, step=1, start=64, end=1024)
NUM_EPOCHS = pn.widgets.IntInput(name='Number of Epochs', value=20, step=1, start=1, end=1000)
SAVE_PLOTS_EPOCH = pn.widgets.IntInput(name='Save Plots After ___ Epochs', value=1, step=1, start=1, end=1000)
SAVE_MODEL_EPOCH = pn.widgets.IntInput(name='Save Models After ___ Epochs', value=5, step=1, start=1, end=1000)

# Load dataset and visualize sample
load_button = pn.widgets.Button(name='Load Data', button_type='primary', margin=(53,0,0,10))

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

def handle_clicks_load(clicks):
    if clicks > 0:
        
        train_dataset = getDataset(TRAIN_DIR, int(RESIZE_TO.value), int(RESIZE_TO.value), get_train_transform())
        valid_dataset = getDataset(VALID_DIR, int(RESIZE_TO.value), int(RESIZE_TO.value), get_valid_transform())
        class_names = train_dataset.classes
        num_samples_text = f"Number of training samples: {len(train_dataset)}\nNumber of validation samples: {len(valid_dataset)}"
        random_index = random.randint(0, len(train_dataset) - 1)
        sample_image = visualize_sample(TRAIN_DIR, int(RESIZE_TO.value), random_index)
        
        return pn.Column(
            pn.pane.Matplotlib(sample_image, margin=(0, 0, 0, -11)),  
            pn.pane.Markdown(num_samples_text, width=200), 
            width=200
        )

loadOutput = pn.Column(pn.bind(handle_clicks_load, load_button.param.clicks))

# Define model name input and train button
newModelName = ipywidgets.Text(
    value='model',
    placeholder='Model Name',
    description='New Model Name',
    disabled=False   
)

train_model_button = pn.widgets.Button(name='Train Model', button_type='primary', margin=(53,0,0,10))

def handle_clicks_train_model(clicks):
    if clicks > 0:
        tqdm.visible = True
        class_names = get_class_names_from_annotations('./images/train', './images/test')
        print(f"Detected class names: {class_names}")
        
        train_dataset = getDataset(TRAIN_DIR, int(RESIZE_TO.value), int(RESIZE_TO.value), get_train_transform())
        valid_dataset = getDataset(VALID_DIR, int(RESIZE_TO.value), int(RESIZE_TO.value), get_valid_transform())
        
        [train_loader, valid_loader] = get_loaders(train_dataset, valid_dataset, int(BATCH_SIZE.value), collate_fn)
        class_names = train_dataset.classes
        DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = create_model(num_classes=(len(class_names)))
        
        train_model(model, train_loader, valid_loader, DEVICE, newModelName.value,
                    int(NUM_EPOCHS.value), MODEL_DIR, PLOT_DIR, int(SAVE_MODEL_EPOCH.value), 
                    int(SAVE_PLOTS_EPOCH.value), tqdm(range(0, NUM_EPOCHS.value)),
                    train_loss_mpl)
        pn.Row(side,output).save('train.html')
        return

# Add the custom CSS for the vertical divider
pn.config.raw_css.append("""
    .vertical-divider {
        border-left: 1px solid #ccc;
        height: 100vh;
        margin-left: 90px;  /* Move the divider to the right */
        margin-top: -15px
    }
""")

# Define the vertical divider
divider = pn.pane.HTML("<div class='vertical-divider'></div>", width=20)    
    
tModelOutput = pn.Column(pn.bind(handle_clicks_train_model, train_model_button.param.clicks))
outrow1 = pn.Row(text2, load_button)
# Organize the layout
output1 = pn.Column(outrow1, loadOutput, width=500)
outrow2 = pn.Row(text3, train_model_button)
output2 = pn.Column(outrow2, train_loss_mpl, tqdm, tModelOutput, margin=(0, 0, 0, 100))
output = pn.Row(output1, divider, output2, sizing_mode="stretch_width")  # Set appropriate max width

accordion = pn.Accordion(
    ("Batch Size", pn.Column(batch_size_description, BATCH_SIZE)),
    ("Image Downsampling Size", pn.Column(resize_to_description, RESIZE_TO)),
    ("Number of Epochs", pn.Column(num_epochs_description, NUM_EPOCHS)),
    ("Save Loss Plots Every _ Epochs", pn.Column(save_plots_epoch_description, SAVE_PLOTS_EPOCH)),
    ("Save Model Every _ Epochs", pn.Column(save_model_epoch_description, SAVE_MODEL_EPOCH)),
    ("Model Name", pn.Column(new_model_name_description))
)
# Define the sidebar with explanations

side = pn.Column(
    text1, 
    pn.Accordion(("Batch Size", pn.Column(batch_size_description))),
    BATCH_SIZE,
    pn.Accordion(("Image Downsampling Size", resize_to_description)),
    RESIZE_TO,
    pn.Accordion(("Number of Epochs", num_epochs_description)),
    NUM_EPOCHS,
    pn.Accordion(("Save Loss Plots Every _ Epochs", save_plots_epoch_description)),
    SAVE_PLOTS_EPOCH,
    pn.Accordion(("Save Model Every _ Epochs", save_model_epoch_description)),
    SAVE_MODEL_EPOCH,
    pn.Accordion(("Model Name", new_model_name_description)),
    newModelName,
)

# Serve the layout
pn.template.MaterialTemplate(
    site="EZ-FRCNN",
    title="Training",
    sidebar=[side],
    main=[output],
).servable();
