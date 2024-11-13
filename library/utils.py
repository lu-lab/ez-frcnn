import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import glob as glob
import xml.etree.ElementTree as et
import csv
import numpy as np
import os

def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))

class getDataset(Dataset):
    def __init__(self, dir_path, width, height, transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = self.get_classes_from_annotations()
        
        
        image_extensions = ['jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp', 'tif']
        all_extensions = image_extensions + [ext.upper() for ext in image_extensions]  # Add uppercase versions
        self.image_paths = glob.glob(f"{self.dir_path}/*.png")
        for extension in all_extensions:
            self.image_paths.extend(glob.glob(f"{self.dir_path}/*.{extension}"))
        # Extract just the filenames
        self.all_images = [os.path.basename(image_path) for image_path in self.image_paths]
        
        self.all_images = sorted(self.all_images)
        
    def get_classes_from_annotations(self):
        """
        Parse all XML files in the dataset directory to build a list of unique classes.
        """
        classes = set()
        xml_files = glob.glob(f"{self.dir_path}/*.xml")
        for xml_file in xml_files:
            tree = et.parse(xml_file)
            root = tree.getroot()
            for member in root.findall('object'):
                try:
                    class_name = member.find('class').text
                except:
                    class_name = member.find('label').text
                classes.add(class_name)
        
        # Add 'background' as the first class and sort the rest alphabetically
        return ['background'] + sorted(classes)
    
    def __getitem__(self, idx):
        # capture the image name and the full image path
        image_name = self.all_images[idx]
        #print(image_name)
        image_path = os.path.join(self.dir_path, image_name)
        #print(image_path)
        # read the image
        image = cv2.imread(image_path)
        # convert BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0
        af = image_name.split('.')
        # capture the corresponding XML file for getting the annotations
        annot_filename = af[0] + '.xml'
        
        annot_file_path = self.dir_path + '/' + annot_filename
        
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        
        root = tree.getroot()
        
        # get the height and width of the image
        image_width = image.shape[1]
        image_height = image.shape[0]

        
        # box coordinates for xml files are extracted and corrected for image size given
        for member in root.findall('object'):
            # map the current object name to `classes` list to get...
            # ... the label index and append to `labels` list
            try:
                labels.append(self.classes.index(member.find('class').text))
            except:
                labels.append(self.classes.index(member.find('label').text))
            try:
                # xmin = left corner x-coordinates
                xmin = int(member.find('xmin').text)
            except:
                # xmin = left corner x-coordinates
                xmin = int(member.find('x').text)    
            try:
                # xmax = right corner x-coordinates
                xmax = int(member.find('xmax').text)
            except:
                # xmax = right corner x-coordinates
                xmax = xmin + int(member.find('width').text)  
            try:
                # ymin = left corner y-coordinates
                ymin = int(member.find('ymin').text)
            except:
                # xmin = left corner y-coordinates
                ymin = int(member.find('y').text)   
            try:
                # ymax = right corner x-coordinates
                ymax = int(member.find('ymax').text)
            except:
                # xmin = left corner y-coordinates
                ymax = ymin + int(member.find('height').text)   
            
            # resize the bounding boxes according to the...
            # ... desired `width`, `height`
            xmin_final = (xmin/image_width)*self.width
            xmax_final = (xmax/image_width)*self.width
            ymin_final = (ymin/image_height)*self.height
            ymax_final = (ymax/image_height)*self.height
            
            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
        
        # bounding box to tensor
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # no crowd instances
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        # apply the image transforms
        if self.transforms:
            sample = self.transforms(image = image_resized,
                                     bboxes = target['boxes'],
                                     labels = labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
            
        return image_resized, target
    def __len__(self):
        return len(self.all_images)

def get_loaders(train_dataset, valid_dataset, BATCH_SIZE, collate_fn):
    train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
    )
    valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
    )
    return [train_loader, valid_loader]
