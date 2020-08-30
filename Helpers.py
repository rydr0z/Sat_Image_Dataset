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
    """Shows 5 clipped satellite images.
    Arguments:
        band (int or list(int)): the bands to be displayed selected from 0,1,2,3,4,5,6,7
        cmap (string): the colormap for the plot
        dataset: SatImageDataset containing images
        seed (int): select seed to display same images each time function executed
    """
    
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
    

def load_datasets(flip=False):
    """Runs functions to create normalized train and test datasets.
    Arguments:
        flip (bool, optional): activates random horizontal and vertical flips data augmentation for dataset.
    """
    train_dataset_raw = SatImageDataset(test=False,
                                    colab=True,
                                    flip=False)
    clean_dataset(train_dataset_raw)
    sat_mean = calculate_mean(train_dataset_raw)
    sat_std = calculate_std(train_dataset_raw)

    print("""=============================================
    Mean of each image channel""")
    print(sat_mean)
    print("""=============================================
    Standard deviation of each image channel""")
    print(sat_std)

    train_dataset_model = SatImageDataset(test=False,
                                          colab=True,
                                          normalize=True,
                                          mean=sat_mean,
                                          std=sat_std,
                                          flip=flip)
    clean_dataset(train_dataset_model)

    print("""
    =============================================Training Set Images: """ + \
     str(train_dataset_model.n_images))

    test_dataset = SatImageDataset(test=True,
                                   colab=True,
                                   normalize=True,
                                   mean=sat_mean,
                                   std=sat_std)
    clean_dataset(test_dataset)
    print("""=============================================
    Test Set Images: """ + str(test_dataset.n_images))
    
    return train_dataset, test_dataset

def plot_results(results_dict, figname="Figure", accuracy=False):
    """ Plot training log results from saved dict, must include Training and Validation Loss and Learning Rate
    for all variants. Optional Training and Validation Accuracy for Classification Result.
    Arguments:
        results_dict (dict): saved log from training run
        figname (string): Top title for figure
        accuracy (bool): True if want to plot accuracy
    """
    if accuracy:                                
        fig, axs = plt.subplots(1,3, figsize=(20,5))
        i = 1
    else:
        fig, axs = plt.subplots(1,2, figsize=(20,5))
        i = 0

    plt.suptitle(figname, y=1.1)
    axs[0].plot(results_dict["Training Loss"], label="Training Loss")
    axs[0].plot(results_dict["Validation Loss"], label="Validation Loss")
    axs[0].axvline(results_dict["Validation Loss"].index(min(results_dict["Validation Loss"])), label="Lowest Val Loss", linestyle=':')
    axs[0].set_title("Model Loss over Training Epochs")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    
    if accuracy:
        axs[i].plot(results_dict["Training Accuracy"], label="Training Accuracy")
        axs[i].plot(results_dict["Validation Accuracy"], label="Validation Accuracy")
        axs[i].axvline(results_dict["Validation Accuracy"].index(max(results_dict["Validation Accuracy"])), 
                 label="Highest Val Acc", linestyle=':')
        axs[i].set_title("Model Accuracy over Training Epochs")
        axs[i].set_xlabel("Epoch")
        axs[i].set_ylabel("Accuracy")
        axs[i].legend()

    axs[i+1].plot(results_dict["Learning Rate"], label="Training Accuracy")
    axs[i+1].set_title("Learning Rate over Training Epochs")
    axs[i+1].set_xlabel("Epoch")
    axs[i+1].set_ylabel("Learning Rate")
    plt.show()

    print("""Lowest Validation Loss: {:.4f} - Epoch: {}\nHighest Validation Accuracy: {:.4f}% - Epoch {}""".format(
    min(results_dict["Validation Loss"]),
    results_dict["Validation Loss"].index(min(results_dict["Validation Loss"])),
    max(results_dict["Validation Accuracy"])*100,
    results_dict["Validation Accuracy"].index(max(results_dict["Validation Accuracy"])))
    )
    
def test_classification(model, dataset, num_images, device):
    from sklearn.metrics import r2_score
    pred_actual_list = []
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1,
                                         shuffle=False)
    iter_loader = iter(loader)
    model.to(device)
    for i in range(0, num_images):
        images, labels = iter_loader.next()
        images = images.float()
        images = images.to(device)
        labels = labels.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(images)
        pred_val1, pred_lab1 = torch.max(outputs, dim=1)
        pred_val2, pred_lab2 = torch.max(outputs[outputs != pred_val1], dim=0)
        outputs1 = outputs[outputs != pred_val1]
        pred_val3, pred_lab3 = torch.max(
            outputs1[outputs1 != pred_val2], dim=0)
        pred = int(pred_lab1.item())
        pred2 = int(pred_lab2.item())
        pred3 = int(pred_lab3.item())
        pop = labels[0][2].item()
        index = labels[0][0].item()
        pred_actual_list.append([index, pred, pop, pred2, pred3])

    indices = [item[0] for item in pred_actual_list]
    preds = [item[1] for item in pred_actual_list]
    pop = [item[2] for item in pred_actual_list]
    preds2 = [item[3] for item in pred_actual_list]
    preds3 = [item[4] for item in pred_actual_list]

    pop_np = np.array(pop)
    preds_np = np.array(preds)
    preds2_np = np.array(preds2)
    preds3_np = np.array(preds3)

    #
    accuracy = ((preds_np == pop_np).sum() / preds_np.size) * 100
    accuracy2 = (np.logical_or((preds_np == pop_np),
                               (preds2_np == pop_np))).sum() / preds2_np.size * 100
    accuracy3 = (np.logical_or(np.logical_or((preds_np == pop_np),
                                             (preds2_np == pop_np)), (preds3_np == pop_np))).sum() / preds3_np.size * 100

    r_squared = r2_score(pop, preds)

    print("""Image Index: {}
    \nModel Predictions: {}
    \nActual Populations: {}
    \nOverall Accuracy (Top 1): {:2f}%
    \nOverall Accuracy (Top 2): {:2f}%
    \nOverall Accuracy (Top 3): {:2f}%
    \nR^2: {:4f}""".format(
        indices, preds, pop, accuracy, accuracy2, accuracy3, r_squared)
    )
    return pred_actual_list