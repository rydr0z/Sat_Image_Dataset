#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[15]:


import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image

import geopandas as gpd
from osgeo import gdal
import numpy as np
import math
import os


# # Helper Functions

# In[2]:


def random_crop_image(image, crp_h=33, crp_w=50):
    '''Return a randomly cropped image with dimensions
    [channels x crp_h x crp_w] given an image with dimensions
    [channels x h x w] where h > crp_h and w > crp_w
    '''
    shape = image.shape
    img_h, img_w = shape[1], shape[2]
    
    
    diff_h = img_h - crp_h
    diff_w = img_w - crp_w
    
    if diff_h > 0:
        rand_pixel_h = np.random.randint(0, diff_h)
    else:
        rand_pixel_h = 0
            
    if diff_w > 0:
        rand_pixel_w = np.random.randint(0, diff_w)
    else:
        rand_pixel_w = 0

    cropped_image = image[:, 
                          rand_pixel_h:crp_h+rand_pixel_h,
                          rand_pixel_w:crp_w+rand_pixel_w]

    return cropped_image


# In[51]:


def horizontal_flip(image):
    image = torch.flip(image, [1])
    return image


# In[52]:


def vertical_flip(image):
    image = torch.flip(image, [2])
    return image


# In[3]:


def classify(label):
    bounds = [2**0, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14, 2**15, 2**16, 2**17]
    for c, bound in enumerate(bounds):
        if label[1] < bound:
            label = torch.cat((label, torch.tensor([[c]])), dim=0)
            break
    return label


def load_images_and_labels(test=False, colab=False):
    '''
    Function for loading TIF images and corresponding shapefiles with population labels.
    '''
    
    if test==False:
    # File Formats and Prefixes
        if colab==False:
            sat_image_folder = 'C:/Users/rdroz/Documents/Dissertation Data Files/Train/Images'
            labels_folder = 'C:/Users/rdroz/Documents/Dissertation Data Files/Train/Labels'
        else:
            sat_image_folder = '/content/Sat_Image_Dataset/Train/Images'
            labels_folder =  '/content/Sat_Image_Dataset/Train/Labels'
        image_prefix = 'Train Set clipped_Index_'

    else:
        if colab==False:
            sat_image_folder = 'C:/Users/rdroz/Documents/Dissertation Data Files/Test/Images'
            labels_folder = 'C:/Users/rdroz/Documents/Dissertation Data Files/Test/Labels'
        else:
            sat_image_folder = '/content/Sat_Image_Dataset/Test/Images'
            labels_folder =  '/content/Sat_Image_Dataset/Test/Labels'
        image_prefix = 'Test Set clipped_Index_'
    
    shapefile_list = os.listdir(labels_folder)
    image_type = '.tif'
    label_prefix = 'Index_'
    label_type = '.gpkg'
        
    # Empty lists for storing information
    image_list = []
    label_list = []
    geo_list = []

    # Loop for loading each image and label
    for sf in shapefile_list:

        # get index number from shapefile name
        i = sf.lstrip('Index_').rstrip('.gpkg')

        # Open image, convert to tensor and crop image, append to list
        gdal_data = gdal.Open(sat_image_folder+'/'+image_prefix+i+image_type)
        image = gdal_data.ReadAsArray()
        image = torch.tensor(image)
        image_list.append(image)
        gdal_data = None

        # Open shapefile and get label and geometry, append to lists
        shp = gpd.read_file(labels_folder+'/'+label_prefix+i+label_type)
        label = np.array(shp['Population'])
        index = shp['Index']
        geometry = shp['geometry']
        label = torch.tensor([index, label])
        label = classify(label)
        geometry = [index, geometry]
        label_list.append(label)
        geo_list.append(geometry)
        shp = None

    return image_list, label_list, geo_list


# In[4]:


def normalize_fn(x, mean, std):
    for i in range(0, len(x)):
        for channel in range(0,x[i].shape[0]):
            x[i][channel] = (x[i][channel] - mean[0][channel])/std[0][channel]


# # Create custom Dataset Class

# In[5]:


class SatImageDataset(Dataset):

    def __init__(self, test=False, colab=False, normalize=False, mean=None, std=None, flip=False):
        # data loading
        self.x, self.y, self.geo = load_images_and_labels(test=test, colab=colab)
        if normalize==True:
            normalize_fn(self.x, mean, std)
        self.n_images = len(self.x)
        self.flip = flip
            
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        self.x[index] = random_crop_image(self.x[index])
        
        if self.flip == True:
            if torch.rand(1) < 0.25:
                self.x[index] = horizontal_flip(self.x[index])
                
            if torch.rand(1) < 0.25:
                self.x[index] = vertical_flip(self.x[index])
                
        return self.x[index], self.y[index]
        
    def __len__(self):
        return self.n_images


# ## Test Dataset Class

# sat_mean = torch.tensor(
#     [[[[486.7721]], [[700.7301]], [[608.2152]], [[2999.1746]], [[1773.8333]],
#       [[2874.8039]], [[975.5184]]]],
#     dtype=torch.float64)

# sat_std = torch.tensor(
#     [[[[192.2344]], [[205.4212]], [[269.9312]], [[961.5902]], [[390.0145]],
#       [[197.9348]], [[303.7064]]]],
#     dtype=torch.float64)

# first = dataset[2]

# image, label = first

# print(image)

# print(label)

# image[torch.isnan(image)] = 0

# print(image)

# image.shape

# print(type(image), type(label))

# plt.imshow(image.permute(2, 1, 0)[:,:,0:3]/3000)

# ## Test DataLoader:
# Visualizing one image and label at a time

# dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

# dataiter = iter(dataloader)

# data = dataiter.next()
# image_iter, labels_iter = data
# print(image_iter.shape)
# plt.imshow(image_iter[0].permute(2, 1, 0)[:,:,3])
# print(labels_iter)
