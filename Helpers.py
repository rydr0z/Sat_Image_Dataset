import numpy as np
import torch
import matplotlib.pyplot as plt



def clean_dataset(dataset):
    NA_sum = 0
    for i in range(0, dataset.n_images):
        NA_sum += torch.sum(torch.isnan(dataset.x[i]))
        dataset.x[i][torch.isnan(dataset.x[i])] = 0
    print("NAN pixel values found and replaced:" + str(NA_sum.item()))
  
  
def calculate_mean(dataset):
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=len(dataset),
                                         shuffle=True)
    images, labels = iter(loader).next()
    mean = torch.mean(images, dim=(0,2,3), keepdim=True)
    
    return mean
    
    
def calculate_std(dataset):
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=len(dataset),
                                         shuffle=True)
    images, labels = iter(loader).next()
    std = torch.std(images, dim=(0,2,3), keepdim=True)
    return std
    
def show_images(band, cmap):
    fig = plt.figure(figsize=(33, 50))
    columns = 5
    rows = 1

    ax = []
    plt.rcParams.update({'font.size': 16})
    for i in range(1, columns * rows + 1):
        image, label = vis_images[i-1], vis_labels[i-1]
        ax.append(fig.add_subplot(rows, columns, i))
        ax[-1].set_title("Image {i} - Population: ".format(i=i) + str(label[1].item()))
        if band != 5:
            plt.imshow(image[0].permute(2, 1, 0)[:, :, band]/3000, cmap=cmap)
        else:
            plt.imshow(image[0].permute(2, 1, 0)[:, :, band]/3000, cmap=cmap)
    plt.show()
    plt.rcParams.update({'font.size': 12})
    
    
def load_images_and_labels(dataset, num_images):
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=num_images,
                                         shuffle=True)
    images, labels = iter(loader).next()
    return images, labels