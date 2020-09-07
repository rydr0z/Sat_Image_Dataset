import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sn


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


def load_datasets(flip=False, calculate_stats=False, classes=16):
    """Runs functions to create normalized train and test datasets.
    Arguments:
        flip (bool, optional): activates random horizontal and vertical flips data augmentation for dataset.
    Returns:
        training dataset, test dataset, channel means, channel standard deviations
    """
    import sys
    sys.path.append('/content/Sat_Image_Dataset')
    from DatasetandDataLoader import SatImageDataset
    if calculate_stats:
        train_dataset_raw = SatImageDataset(test=False,
                                            colab=True,
                                            flip=False,
                                            classes=classes)
        clean_dataset(train_dataset_raw)
        sat_mean = calculate_mean(train_dataset_raw)
        sat_std = calculate_std(train_dataset_raw)
    else:
        sat_mean = torch.tensor([[[[ 481.4347]],

         [[ 661.0318]],

         [[ 633.2782]],

         [[2610.5615]],

         [[1754.3627]],

         [[2808.4644]],

         [[ 987.0544]]]], dtype=torch.float64)
        
        sat_std = torch.tensor([[[[182.0427]],

         [[221.7736]],

         [[282.7514]],

         [[989.6967]],

         [[604.5539]],

         [[463.1821]],

         [[403.9931]]]], dtype=torch.float64)

    print("""=============================================\
    \nMean for image channels:""")
    print(sat_mean)
    print("""=============================================\
    \nStandard deviation for image channels:""")
    print(sat_std)

    train_dataset_model = SatImageDataset(test=False,
                                          colab=True,
                                          normalize=True,
                                          mean=sat_mean,
                                          std=sat_std,
                                          flip=flip,
                                          classes=classes)
    clean_dataset(train_dataset_model)

    print("""=============================================\
    \nTraining Set Images: """ + \
     str(train_dataset_model.n_images))

    test_dataset = SatImageDataset(test=True,
                                   colab=True,
                                   normalize=True,
                                   mean=sat_mean,
                                   std=sat_std,
                                   classes=classes)
    clean_dataset(test_dataset)
    print("""=============================================\
    \nTest Set Images: """ + str(test_dataset.n_images))
    
    return train_dataset_model, test_dataset, sat_mean, sat_std

def plot_results_cl(results_dict, figname="Figure"):
    """ Plot training log results from saved dict, must include Training and Validation Loss and Learning Rate
    for all variants. Optional Training and Validation Accuracy for Classification Result.
    Arguments:
        results_dict (dict): saved log from training run
        figname (string): Top title for figure
        accuracy (bool): True if want to plot accuracy
    """                               
    fig, axs = plt.subplots(1,2, figsize=(30,5))

    plt.suptitle(figname, y=1.1)
    axs[0].plot(results_dict["Training Loss"], label="Training Loss")
    axs[0].plot(results_dict["Validation Loss"], label="Validation Loss")
    axs[0].axvline(results_dict["Validation Loss"].index(min(results_dict["Validation Loss"])), label="Lowest Val Loss", linestyle=':')
    axs[0].set_title("Model Loss over Training Epochs")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    
    axs[1].plot(results_dict["Training Accuracy"], label="Training Accuracy")
    axs[1].plot(results_dict["Validation Accuracy"], label="Validation Accuracy")
    axs[1].axvline(results_dict["Validation Accuracy"].index(max(results_dict["Validation Accuracy"])), 
             label="Highest Val Acc", linestyle=':')
    axs[1].set_title("Model Accuracy over Training Epochs")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    plt.show()
    

    print("""Lowest Validation Loss: {:.4f} - Epoch: {}\nHighest Validation Accuracy: {:.4f}% - Epoch {}""".format(
        min(results_dict["Validation Loss"]),
        results_dict["Validation Loss"].index(min(results_dict["Validation Loss"])),
        max(results_dict["Validation Accuracy"])*100,
        results_dict["Validation Accuracy"].index(max(results_dict["Validation Accuracy"])))
         )
    
def plot_results_reg(results_dict, figname="Figure"):
    """ Plot training log results from saved dict, must include Training and Validation Loss and Learning Rate
    for all variants. Optional Training and Validation Accuracy for Classification Result.
    Arguments:
        results_dict (dict): saved log from training run
        figname (string): Top title for figure
        accuracy (bool): True if want to plot accuracy
    """                              
    fig, axs = plt.subplots(1,2, figsize=(30,5))

    plt.suptitle(figname, y=1.1)
    axs[0].plot(results_dict["Training Loss"], label="Training Loss")
    axs[0].plot(results_dict["Validation Loss"], label="Validation Loss")
    axs[0].axvline(results_dict["Validation Loss"].index(min(results_dict["Validation Loss"])), label="Lowest Val Loss", linestyle=':')
    axs[0].set_title("Model Loss over Training Epochs")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    
    axs[1].plot(results_dict["Training RMSE"], label="Training RMSE")
    axs[1].plot(results_dict["Validation RMSE"], label="Validation RMSE")
    axs[1].axvline(results_dict["Validation RMSE"].index(min(results_dict["Validation RMSE"])), 
             label="Lowest Val RMSE", linestyle=':')
    axs[1].set_title("Model RMSE over Training Epochs")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("RMSE")
    axs[1].legend()
    plt.show()

    print("""Lowest Validation Loss: {:.4f} - Epoch: {}\nLowest RMSE: {:.4f} - Epoch {}""".format(
        min(results_dict["Validation Loss"]),
        results_dict["Validation Loss"].index(min(results_dict["Validation Loss"])),
        min(results_dict["Validation RMSE"]),
        results_dict["Validation RMSE"].index(min(results_dict["Validation RMSE"])))
         )
            
def test_classification(model, dataset, num_images, device):
    from sklearn.metrics import classification_report
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

    cl_report = classification_report(pop, preds)

    print("""Image Index: {}
    \nModel Predictions: {}
    \nActual Populations: {}
    \nOverall Accuracy (Top 1): {:2f}%
    \nOverall Accuracy (Top 2): {:2f}%
    \nOverall Accuracy (Top 3): {:2f}%
    \n
    \n{}""".format(
        indices, preds, pop, accuracy, accuracy2, accuracy3, cl_report)
    )
    return pred_actual_list

def test_regression(model, dataset, num_images, device):
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, \
    explained_variance_score

    pred_actual_list = []
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1,
                                         shuffle=False)
    iter_loader = iter(loader)
    model.to(device)
    for i in range(0, num_images):
        images, labels = iter_loader.next()
        images = images.float()
        images, labels = images.to(device), labels.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(images)
        pred = int(outputs[0].item())
        pop = int(labels[0][1].item())
        index = int(labels[0][0].item())
        pred_actual_list.append([index, pred, pop])

    indices = [item[0] for item in pred_actual_list]
    preds = [item[1] for item in pred_actual_list]
    pop = [item[2] for item in pred_actual_list]

    r_squared = r2_score(pop, preds)
    mae = mean_absolute_error(pop, preds)
    rmse = mean_squared_error(pop, preds, squared=False)
    exp_var = explained_variance_score(pop, preds)

    sum_pop = np.sum(np.array(pop))
    sum_pred = np.sum(np.array(preds))

    print("""Actual Total Population for Test Set Regions: {}
    Predicted Total Population for Test Set Regions: {}
          \nMAE: {:.2f}
          \nRMSE: {:.2f}
          \nR^2: {:.4f}
          \nExplained Variance: {:.4f}""".format(sum_pop, sum_pred, mae, rmse, r_squared, exp_var))
    return pred_actual_list