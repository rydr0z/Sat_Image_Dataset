import torch
from torch.utils.data import Dataset, DataLoader

import geopandas as gpd
from osgeo import gdal
import numpy as np
import os

### Setup for your folder location on local device and Google Colab Storage
class SatImageDataset(Dataset):
    
    def __init__(self,
                 test=False,
                 colab=False,
                 normalize=False,
                 flip=False,
                 flip_prob=0.25,
                 classes=16,
                 n_to_load=None):
        """Creates SatImageDataset
        Arguments:
            test = True for test set, False for training set
            colab = True If running code on Google Colab
            normalize = True if want images to be normalized
            mean = mean for each image channel, optional
            std = standard deviation for each image channel, optional
            flip = True for random horizontal and vertical flips
            flip_prob = Probability of flipping an image
            classes = either 16 or 6 depending on problem
        Returns:
            (image_list , label_list, geo_list)"""
        
        self.test = test
        self.colab = colab
        self.classes = classes
        self.n_to_load = n_to_load
        
        # Folder options for loading data
        self.local_folder = 'C:/Users/rdroz/Documents/GitHub/Sat_Image_Dataset/Dataset/'
        self.colab_folder = '/content/Sat_Image_Dataset/'
                
        # Variables that will be used in loading data
        self.bounds_16class = [2**0, 2**1, 2**2, 2**3, 2**4,2**5, 2**6, 2**7,
                               2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14]
        self.bounds_6class = [1, 10, 100, 1000, 10000]
        
        # Optional data transformation flags
        self.flip = flip
        self.flip_prob = flip_prob
        self.normalize = normalize
        
        # data loading
        self.x, self.y, self.geo = self.load_images_and_labels()
        
        self.n_images = len(self.x)
        self.mean = torch.mean(torch.stack(self.x), dim=(0,2,3), keepdim=True)
        self.std = torch.std(torch.stack(self.x), dim=(0,2,3), keepdim=True)

        

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        im = self.x[index]
        
        if self.normalize == True:
            im = self.normalize_fn(im)

        if self.flip == True:
            if torch.rand(1) < self.flip_prob:
                im = self.horizontal_flip(im)
            elif torch.rand(1) < self.flip_prob:
                im = self.vertical_flip(im)

        return im, self.y[index]

    def __len__(self):
        return self.n_images
    
    def normalize_fn(self, image):
        '''Normalizes an image using channel means and channel standard deviations.
        Changes are made in place.
        Arguments:
        x - image to be normalized
        mean - calculated image mean by channel
        std - caluclated image standard deviation by channel'''
        img = image
        for channel in range(0, img.shape[0]):
            img[channel] = \
                (img[channel] - self.mean[0][channel]) / self.std[0][channel]
        return img
    
    def horizontal_flip(self, image):
        '''Flips an image horizontally'''
        image = torch.flip(image, [1])
        return image


    def vertical_flip(self, image):
        '''Flips an image vertically'''
        image = torch.flip(image, [2])
        return image
    
    def random_crop_image(self, image, crp_h=33, crp_w=50):
        '''Return a randomly cropped image with dimensions
        [channels x crp_h x crp_w] given an image with dimensions
        [channels x h x w] where h > crp_h and w > crp_w.
        Arguments:
            image (tensor): image to be cropped
            crp_h (int): height of cropped image
            crp_w (int): width of cropped image
        Returns:
            image (tensor) with dimensions c x h x w
        '''
        img_h, img_w  = image.shape[1], image.shape[2] # get image shape
        
        # Find the difference between actual height and desired crop
        diff_h = img_h - crp_h
        diff_w = img_w - crp_w
        
        # If difference is positive, choose random height and width pixels 
        # indices to start cropping
        if diff_h > 0:
            rand_pixel_h = np.random.randint(0, diff_h)
        else:
            rand_pixel_h = 0
    
        if diff_w > 0:
            rand_pixel_w = np.random.randint(0, diff_w)
        else:
            rand_pixel_w = 0
        
        # Using selected indices, crop the image
        cropped_image = image[:,
                              rand_pixel_h:crp_h + rand_pixel_h,
                              rand_pixel_w:crp_w + rand_pixel_w]
    
        return cropped_image
    
    def create_classes_from_population(self, label):
        '''Create class labels from population labels
        Arguments:
            label - label to create a class for
            classes - int, either 16 or 6 classes
            16: bounds are 2**0, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14
            6: bounds are 1, 10, 100, 1000, 10000
        Returns:
            label with added dimension with class label
        '''
        if self.classes == 16:
            bounds = self.bounds_16class
        if self.classes == 6:
            bounds = self.bounds_6class
            
        if label[1] > bounds[-1]:
            label = torch.cat((label, torch.tensor([[len(bounds)]], dtype=torch.double)), dim=0)
        else:
            for c, bound in enumerate(bounds):
                if label[1] < bound:
                    label = torch.cat((label, torch.tensor([[c]], dtype=torch.double)), dim=0)
                    break
        return label
    
    def load_images_and_labels(self):
        '''Function for loading TIF images and corresponding shapefiles with population labels.
        Arguments:
            test - True for loading test set, False for loading training set
            colab - True if loading from google colab storage
            classes - 16 or 6 depending on desired classification bounds
            n_to_load - int: how many examples to load
        Returns:
            (image_list , label_list, geo_list)
            
        '''        
        # File Formats and Prefixes
        if self.test:
            test_or_train = 'Test'
        else:
            test_or_train = 'Train'
            
        if self.colab:
            sat_image_folder = self.colab_folder+test_or_train+'/Images'
            labels_folder = self.colab_folder+test_or_train+'/Labels'
        else:
            sat_image_folder = self.local_folder+test_or_train+'/Images'
            labels_folder = self.local_folder+test_or_train+'/Labels'
        
        image_prefix = test_or_train+' Set clipped_Index_'
    
        shapefile_list = os.listdir(labels_folder)
        if self.n_to_load is not None:
            shapefile_list = shapefile_list[0:self.n_to_load]
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
            gdal_data = gdal.Open(sat_image_folder + '/' + image_prefix + i +
                                  image_type)
            image = gdal_data.ReadAsArray()
            image = torch.tensor(image)
            image = self.random_crop_image(image)  # Crops the tensor to desired size of (33 x 50)
            image = torch.where(torch.isnan(image),
                                torch.zeros_like(image),
                                image) # replaces NaN pixel values with 0
            
            image_list.append(image)
            gdal_data = None
    
            # Open shapefile and get label and geometry, append to lists
            shp = gpd.read_file(labels_folder + '/' + label_prefix + i +
                                label_type)
            label = np.array(shp['Population'])
            index = shp['Index']
            geometry = shp['geometry']
            label = torch.tensor([index, label])
            label = self.create_classes_from_population(label)
            geometry = [index, geometry]
            label_list.append(label)
            geo_list.append(geometry)
            shp = None
    
        return image_list, label_list, geo_list

def sat_image_dataset_tests():
    test_dataset = SatImageDataset(test=False,colab=False,normalize=True,n_to_load=1000)
    
    ### Normalization Test
    image_list = []
    # Fetching all items in dataset then calculating the mean and standard deviation
    # after normalization
    for i in range(0,test_dataset.__len__()):
        image_list.append(test_dataset.__getitem__(i)[0])
    mean = torch.mean(torch.stack(image_list), dim=(0,2,3), keepdim=True)
    std = torch.std(torch.stack(image_list), dim=(0,2,3), keepdim=True)
    
    # Check that all channel means are close to zero
    assert np.abs(mean.sum(dim=0)[0].item()) < 1e-10, \
       'Calculated mean after normalization is not equal to 0 for all channels'
       
    # Check that  all channel standard deviations are approximately one
    assert np.abs(std.sum().item() - (1 * std.shape[1])) < 1e-10, \
        '''Calculated standard deviation after normalization is not equal
        to 1 for all channels'''
    
sat_image_dataset_tests()
