import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import cv2
import os
from PIL import Image
from torchvision.io import read_image 
from data.imgaug import GetTransforms
from data.utils import transform
np.random.seed(0)
import sys 
sys.path.append('/media/Datacenter_storage/')


class ImageDataset_Mayo(Dataset):
    def __init__(self, label_path, cfg, mode='train'):
        self.cfg = cfg
        self._label_header = None
        self._bias_header = None
        self._image_paths = []
        self._image_paths_inverted = []
        self._bias = []
        self._causal = []
        self._causal2 = []
        self._labels = []
        self._mode = mode
        # Define the indices of the columns you want to extract
        label_columns_indices = [5,6,7,8]  # MACE_6mo, MACE_1yr, MACE_2yr, MACE_5yr
        #label_columns_indices = [3,4,5,6] 
        #bias_columns_indices = [7,8,9,10,11,12]
        ##CKD, Chf, Diabetes, Hpyertension
        #causal_columns_indices = [11,12,13,14]
        ##CKD and Chf
        causal_columns_indices = [11,14]
        ##Chf
        #causal_columns_indices = [11]
        #causal group 2: Diabetes and Hypertension
        causal2_columns_indices = [12,13]
        bias_columns_indices = [9,10]
        with open(label_path) as f:
            header = f.readline().strip('\n').split(',')
            self._label_header = [header[i] for i in label_columns_indices]
            self._bias_header = [header[i] for i in bias_columns_indices]
            self._causal_header = [header[i] for i in causal_columns_indices]
            self._causal2_header = [header[i] for i in causal2_columns_indices]
            for line in f:
            #for i, line in enumerate(f):
                # Check if we've reached the 100th line
                #if i >= 20:
                  #  break  # Exit the loop
                fields = line.strip('\n').split(',')
                image_path = fields[1][1:]
                print(f"image_path: {image_path}")
                if fields[3] == 'MONOCHROME1':
                #if fields[2] == 'MONOCHROME1':
                    image_inverted = fields[1][1:]
                    self._image_paths_inverted.append(image_inverted)
                labels = [fields[i] for i in label_columns_indices]
                causal = [fields[i] for i in causal_columns_indices]
                causal2 = [fields[i] for i in causal2_columns_indices]
                bias = [fields[i] for i in bias_columns_indices]
                #print(f"bias:{bias}")
                self._image_paths.append(image_path)
                self._labels.append(labels)
                self._bias.append(bias)
                self._causal.append(causal)
                self._causal2.append(causal2)
                
            
        self._num_image = len(self._image_paths)

    def __len__(self):
        return self._num_image
    
    current_directory = os.getcwd()

    print("The script is running from:", current_directory)


    
    def __getitem__(self, idx):
        #print(f"image_path: {self._image_paths[idx]}")
        image_path = self._image_paths[idx]
        image = cv2.imread(image_path, 0)  # Load the current image in grayscale

        # Check if this image needs to be inverted
        if image_path in self._image_paths_inverted:
            # This image needs to be inverted
            inverted_image_array = 255 - image
            image = inverted_image_array  # Now 'image' is the inverted image
        

        # Convert to PIL Image for both inverted and non-inverted images
        folder_name = "/home/jialu/Datacenter_storage/jialu_/jialu_causalv2/saved_images_invert"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        image = Image.fromarray(image)

        save_indices = [0, 100, 1000, 5000, 8000, 10000, 12000]
        # Save the image right after conversion to PIL Image if it's one of the specified indices
      #  save_indices = [0, 100, 1000, 5000, 8000, 10000, 12000]
        if idx in save_indices:
            save_path_early = os.path.join(folder_name, f"early_saved_image_{idx}.png")
            image.save(save_path_early)

        # Apply transformations if in 'train' mode
        if self._mode == 'train':
            image = GetTransforms(image, type=self.cfg.use_transforms_type)
        
        # Convert the PIL Image back to a NumPy array for further processing if necessary
        image = np.array(image)
        #print(f"image2:{image.shape}")
        #image_pil = Image.fromarray(image.astype(np.uint8))
        #save_path = os.path.join(folder_name, f"saved_image2_{idx+1}.png")
        # Define the indices at which you want to save the images
        #save_indices = [0, 100, 1000, 5000, 8000, 10000, 12000]

        # Apply any additional transformations or processing
        image = transform(image, self.cfg)
       # if idx in save_indices:
            
            # Convert the numpy array back to a PIL Image
        #    image_pil = Image.fromarray(image_beforenorm.astype(np.uint8))
            
            # Define the save path with the current index to differentiate the images
        #    save_path = os.path.join(folder_name, f"saved_image_{idx}.png")
        #    image_pil.save(save_path)
        

        #print(f"image3:{image.shape}")
        labels = np.array(self._labels[idx]).astype(np.float32)
        bias = np.array(self._bias[idx]).astype(np.float32)
        causal = np.array(self._causal[idx]).astype(np.float32)
        causal2 = np.array(self._causal2[idx]).astype(np.float32)
        path = self._image_paths[idx]
        
        # Structure the return based on the operation mode
        if self._mode in ['train', 'dev']:
            return image, labels, bias, causal, 
        elif self._mode in ['test', 'heatmap']:
            return image, path, bias, labels, causal
        else:
            raise Exception(f'Unknown mode: {self._mode}')

        
        



