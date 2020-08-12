import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage import exposure



def clean_dataset(dataset):
    NA_sum = 0
    for i in range(0, dataset.n_images):
        NA_sum += torch.sum(torch.isnan(dataset.x[i]))
        dataset.x[i][torch.isnan(dataset.x[i])] = 0
    NA_percent = NA_sum.item() / ( dataset.n_images * dataset.x[0].shape[1] * dataset.x[0].shape[2])
    print("NAN pixel values found and replaced:" + str(NA_sum.item()) + "\nPercentage of total pixels:" + str(NA_percent))
  
  
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
    
def show_images(band, cmap, dataset, seed):
    
    torch.manual_seed(seed)
    train_loader_for_vis = torch.utils.data.DataLoader(dataset,
                                                   batch_size=1,
                                                   shuffle=True)

    vis_images = []
    vis_labels = []

    data_iter = iter(train_loader_for_vis)
    for i in range(1,6):
        vis_image, vis_label = data_iter.next()
        vis_images.append(vis_image)
        vis_labels.append(vis_label)

    fig = plt.figure(figsize=(33, 50))
    columns = 5
    rows = 1

    ax = []
    plt.rcParams.update({'font.size': 18})
    for i in range(1, columns * rows + 1):
        image, label = vis_images[i-1], vis_labels[i-1]
        ax.append(fig.add_subplot(rows, columns, i))
        ax[-1].set_title("Image {i}\nPopulation: {p}\nClass: {c}".format(i=i, p=str(label[:,1,:].item()), c=str(label[:,2,:].item())))
        if band != 5:
            plt.imshow(image[0].permute(2, 1, 0)[:, :, band]/3000, cmap=cmap, vmin=0, vmax=1)
        else:
            print(torch.max(image[0].permute(2, 1, 0)[:, :, band]))
            plt.imshow(image[0].permute(2, 1, 0)[:, :, band], cmap=cmap)
    plt.show()
    plt.rcParams.update({'font.size': 12})
    plt.savefig("\content\drive\My Drive\Dissertation Files\Exported Figures\Example Images - Band {}".format(band), format="png")
    
    
def load_images_and_labels(dataset, num_images):
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=num_images,
                                         shuffle=True)
    images, labels = iter(loader).next()
    return images, labels