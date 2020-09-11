import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sn


def clean_dataset(dataset):
    """Removes replaces NA value pixels with 0
    Arguments:
        Dataset - to be cleaned, inplace"""
    NA_sum = 0
    for i in range(0, dataset.n_images):
        NA_sum += torch.sum(torch.isnan(dataset.x[i]))
        dataset.x[i][torch.isnan(dataset.x[i])] = 0
    NA_percent = NA_sum.item() / (dataset.n_images * dataset.x[0].shape[1] *
                                  dataset.x[0].shape[2])
    print("NAN pixel values found and replaced:" + str(NA_sum.item()) +
          "\nPercentage of total pixels:" + str(NA_percent))


def calculate_mean(dataset):
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=len(dataset),
                                         shuffle=True)
    images, labels = iter(loader).next()
    mean = torch.mean(images, dim=(0, 2, 3), keepdim=True)

    return mean


def calculate_std(dataset):
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=len(dataset),
                                         shuffle=True)
    images, labels = iter(loader).next()
    std = torch.std(images, dim=(0, 2, 3), keepdim=True)
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
        sat_mean = torch.tensor(
            [[[[481.4347]], [[661.0318]], [[633.2782]], [[2610.5615]],
              [[1754.3627]], [[2808.4644]], [[987.0544]]]],
            dtype=torch.float64)

        sat_std = torch.tensor(
            [[[[182.0427]], [[221.7736]], [[282.7514]], [[989.6967]],
              [[604.5539]], [[463.1821]], [[403.9931]]]],
            dtype=torch.float64)

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
    fig, axs = plt.subplots(1, 2, figsize=(30, 5))

    plt.suptitle(figname, y=1.1)
    axs[0].plot(results_dict["Training Loss"], label="Training Loss")
    axs[0].plot(results_dict["Validation Loss"], label="Validation Loss")
    axs[0].axvline(results_dict["Validation Loss"].index(
        min(results_dict["Validation Loss"])),
                   label="Lowest Val Loss",
                   linestyle=':')
    axs[0].set_title("Model Loss over Training Epochs")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    axs[1].plot(results_dict["Training Accuracy"], label="Training Accuracy")
    axs[1].plot(results_dict["Validation Accuracy"],
                label="Validation Accuracy")
    axs[1].axvline(results_dict["Validation Accuracy"].index(
        max(results_dict["Validation Accuracy"])),
                   label="Highest Val Acc",
                   linestyle=':')
    axs[1].set_title("Model Accuracy over Training Epochs")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    plt.show()

    print(
        """Lowest Validation Loss: {:.4f} - Epoch: {}\nHighest Validation Accuracy: {:.4f}% - Epoch {}"""
        .format(
            min(results_dict["Validation Loss"]),
            results_dict["Validation Loss"].index(
                min(results_dict["Validation Loss"])),
            max(results_dict["Validation Accuracy"]) * 100,
            results_dict["Validation Accuracy"].index(
                max(results_dict["Validation Accuracy"]))))


def plot_results_reg(results_dict, figname="Figure"):
    """ Plot training log results from saved dict, must include Training and Validation Loss and Learning Rate
    for all variants. Optional Training and Validation Accuracy for Classification Result.
    Arguments:
        results_dict (dict): saved log from training run
        figname (string): Top title for figure
        accuracy (bool): True if want to plot accuracy
    """
    fig, axs = plt.subplots(1, 2, figsize=(30, 5))

    plt.suptitle(figname, y=1.1)
    axs[0].plot(results_dict["Training Loss"], label="Training Loss")
    axs[0].plot(results_dict["Validation Loss"], label="Validation Loss")
    axs[0].axvline(results_dict["Validation Loss"].index(
        min(results_dict["Validation Loss"])),
                   label="Lowest Val Loss",
                   linestyle=':')
    axs[0].set_title("Model Loss over Training Epochs")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    axs[1].plot(results_dict["Training RMSE"], label="Training RMSE")
    axs[1].plot(results_dict["Validation RMSE"], label="Validation RMSE")
    axs[1].axvline(results_dict["Validation RMSE"].index(
        min(results_dict["Validation RMSE"])),
                   label="Lowest Val RMSE",
                   linestyle=':')
    axs[1].set_title("Model RMSE over Training Epochs")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("RMSE")
    axs[1].legend()
    plt.show()

    print(
        """Lowest Validation Loss: {:.4f} - Epoch: {}\nLowest RMSE: {:.4f} - Epoch {}"""
        .format(
            min(results_dict["Validation Loss"]),
            results_dict["Validation Loss"].index(
                min(results_dict["Validation Loss"])),
            min(results_dict["Validation RMSE"]),
            results_dict["Validation RMSE"].index(
                min(results_dict["Validation RMSE"]))))


def test_regression(model, dataset, num_images, device):
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, \
    explained_variance_score

    #load images and labels and apply model to each one
    pred_actual_list = []
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
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

    # calculate prediction metrics
    r_squared = r2_score(pop, preds)
    mae = mean_absolute_error(pop, preds)
    rmse = mean_squared_error(pop, preds, squared=False)
    exp_var = explained_variance_score(pop, preds)

    sum_pop = np.sum(np.array(pop))
    sum_pred = np.sum(np.array(preds))
    mean_pop = np.mean(np.array(pop))
    mean_pred = np.mean(np.array(preds))
    percent_mae = (mae / mean_pop) * 100

    print("""Actual Total Population for Region: {}
          \nPredicted Total Population for Region: {}
          \nActual Mean Population for Region: {}
          \nPredicted Mean Population for Region: {}
          \nMAE: {:.2f}
          \n%MAE: {:.2f}
          \nRMSE: {:.2f}
          \nR^2: {:.4f}
          \nExplained Variance: {:.4f}""".format(sum_pop, sum_pred, mean_pop,
                                                 mean_pred, mae, percent_mae,
                                                 rmse, r_squared, exp_var))
    return pred_actual_list


def classes_to_population(results, dataset, classes=16):
    """Converts class predictions to population using midpoint method
    Arguments:
        results = list of results with format [index, pred, pop]
        dataset = SatImageDataset to calculate actual population
        classes = 16 or 6 depending on chosen problem
    Returns:
        new_preds = list of population results"""
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, \
    explained_variance_score

    # Load all necessary data in correct format
    preds = [item[1] for item in results]
    pop = []

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    iter_loader = iter(loader)
    for i in range(0, len(loader)):
        _, labels = iter_loader.next()
        pop_item = int(labels[0][1].item())
        pop.append(pop_item)

    new_preds = []

    # Define class bounds
    if classes == 16:
        bounds = [
            2**0, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10,
            2**11, 2**12, 2**13, 2**14
        ]
    if classes == 6:
        bounds = [1, 10, 100, 1000, 10000]

    # Calculate population values for each class
    for pred in preds:
        if pred == 0:
            new_preds.append(0)
        elif pred == len(bounds):
            new_preds.append(bounds[-1])
        else:
            midpoint = bounds[pred - 1] + bounds[pred]
            midpoint /= 2
            new_preds.append(midpoint)

    # calculate prediction metrics
    r_squared = r2_score(pop, new_preds)
    mae = mean_absolute_error(pop, new_preds)
    rmse = mean_squared_error(pop, new_preds, squared=False)
    exp_var = explained_variance_score(pop, new_preds)

    sum_pop = np.sum(np.array(pop))
    sum_pred = np.sum(np.array(new_preds))
    mean_pop = np.mean(np.array(pop))
    mean_pred = np.mean(np.array(new_preds))
    percent_mae = (mae / mean_pop) * 100

    print("""Actual Total Population for Region: {}
          \nPredicted Total Population for Region: {}
          \nActual Mean Population for Region: {}
          \nPredicted Mean Population for Region: {}
          \nMAE: {:.2f}
          \n%MAE: {:.2f}
          \nRMSE: {:.2f}
          \nR^2: {:.4f}
          \nExplained Variance: {:.4f}""".format(sum_pop, sum_pred, mean_pop,
                                                 mean_pred, mae, percent_mae,
                                                 rmse, r_squared, exp_var))

    return new_preds