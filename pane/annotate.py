import sys
from pathlib import Path
# Add the parent directory to the Python path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
# Now import from 'library'
from library import init_annotations, split_images_and_xml
import panel as pn
import ipywidgets

pn.extension('ipywidgets')
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
# Class input section
classes = ['frog']
textbar = ipywidgets.Text(
    value='Frog',
    placeholder='Type something',
    description='Classes:',
    disabled=False   
)

# Function to initialize annotations
def cb(test):
    return init_annotations(test.replace(" ", "").split(','))

x1 = pn.Row(pn.bind(cb, textbar), sizing_mode='stretch_width', width_policy='auto', height_policy='fit', max_height=500)

# Test ratio and shuffle button section
test_ratio = pn.widgets.FloatInput(name='Test Ratio', value=0.1, step=1e-1, start=0, end=1)
shuffle_button = pn.widgets.Button(name='Shuffle Data', button_type='primary')

# Function to handle shuffle button click
def handle_clicks(clicks):
    if clicks > 0:
        split_images_and_xml(source_folder='./annotations/', test_ratio=float(test_ratio.value))

# Combine shuffle button with handler
but = pn.Column(shuffle_button, pn.bind(handle_clicks, shuffle_button.param.clicks))

# Descriptive text sections
classes_description = pn.pane.Markdown("## 1. Classes Input\nEnter the names of the classes separated by commas.")
# Step 3: Test ratio and shuffle button section with numbered header
test_ratio_description = pn.pane.Markdown("## 3. Shuffle Data and Split Dataset\nSpecify the ratio for splitting the dataset into training and test sets and shuffle the data.")
shuffle_description = pn.pane.Markdown("### Shuffle Data\nAfter all annotations are complete, click the button to shuffle and move data into the training folder.")
# Step 2: Labeling section with numbered header (Placeholder for the labeling tool)
labeling_description = pn.pane.Markdown("## 2. Perform annotations using the labeling tool.")

# Instruction for using the labeling tool
labeling_instructions = pn.pane.Markdown("""
### How to Use the Labeling Tool:
- **Click and drag** to draw a bounding box around the object to annotate.
- **Class selection**: Either click on the class label buttons or press keys **1-10** to select the class for labeling.
- **Reset**: Click "Reset" to clear the current annotations for the image.
- **Submit**: Save the annotations by clicking the "Submit" button.
- **Skip**: Move to the next image without saving the current annotations.
- **Go Back**: Return to the previous image.
""", width=500)

# Layout for sidebar with descriptive sections
side = pn.Column(
    classes_description, textbar,
    pn.Spacer(height=155),  # Spacer to add vertical space between sections
    test_ratio_description, test_ratio,
    shuffle_description, but
)

# Layout for the app with two-column main panel
# Layout for the main panel with bounding box tool and instructions aligned closer
# Layout for the main panel with explicit widths
main_panel = pn.Row(
    pn.Column(labeling_description, x1, width=500),  # Set width for the labeling tool section
    
    pn.Column(pn.Spacer(height=110), labeling_instructions)  # Right: Instructions
)

# Create the MaterialTemplate with the sidebar and main panel
pn.template.MaterialTemplate(
    site="EZ-FRCNN",
    title="Annotator",
    sidebar=[side],
    main=[main_panel],
).servable();
