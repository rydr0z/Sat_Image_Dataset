import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sn

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


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
    fig, axs = plt.subplots(1,3, figsize=(30,5))

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

    axs[2].plot(results_dict["Learning Rate"], label="Training Accuracy")
    axs[2].set_title("Learning Rate over Training Epochs")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Learning Rate")
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
    fig, axs = plt.subplots(1,3, figsize=(30,5))

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

    axs[2].plot(results_dict["Learning Rate"], label="Training Accuracy")
    axs[2].set_title("Learning Rate over Training Epochs")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Learning Rate")
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
    mse = mean_squared_error(pop, preds)
    exp_var = explained_variance_score(pop, preds)

    print("""Image Index: {}
          \nModel Predictions: {}
          \nActual Populations: {}
          \nMAE: {:.2f}
          \nMSE: {:.2f}
          \nR^2: {:.4f}
          \nExplained Variance: {:.4f}""".format(indices, preds, pop, mae, mse, r_squared, exp_var))
    return pred_actual_list

def population_hist(dataset, bins=10, figsize=(20,10)):
    fig, ax = plt.subplots(1,1,figsize=figsize)
    pop_list = []
    for i in range(0, len(dataset)):
        pop = dataset[i][1][1].item()
        pop_list.append(pop)
    bins_list = sn.distplot(pop_list, bins=bins, kde=False, norm_hist=False)
    return bins_list

def class_hist(dataset, figsize=(20, 10), classes=16):
    import matplotlib.ticker as ticker
    bins = range(0, classes+1)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    class_list = []
    for i in range(0, len(dataset)):
        cl = dataset[i][1][2].item()
        class_list.append(cl)
    bins_list = plt.hist(class_list, bins=bins)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(0, classes+1) + 0.5))
    
    if classes == 16:
        ax.xaxis.set_minor_formatter(
            ticker.FixedFormatter([
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
                '13', '14', '15'
            ]))
        
    if classes == 9
        ax.xaxis.set_minor_formatter(
            ticker.FixedFormatter([
                '0', '1', '2', '3', '4', '5', '6', '7', '8'
            ]))
    return bins_list

def population_hist(dataset, bins=10, figsize=(20,10)):
    fig, ax = plt.subplots(1,1,figsize=figsize)
    pop_list = []
    for i in range(0, len(dataset)):
        pop = dataset[i][1][1].item()
        pop_list.append(pop)
    bins_list = sn.distplot(pop_list, bins=bins, kde=False, norm_hist=False)
    return bins_list

def balanced_class_hist(train_loader, figsize=(20, 10), classes=16):
    import matplotlib.ticker as ticker
    bins=range(0, classes+1)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    cl_list = []
    iter_loader_test = iter(train_loader)
    for i in range(0, len(train_loader)):
        image, label = iter_loader_test.next()
        for j in range(0, label.shape[0]):
            cl = label[1][2].item()
            cl_list.append(cl)
    bins_list = plt.hist(cl_list, bins=bins)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(0, classes+1) + 0.5))
    if classes == 16:
        ax.xaxis.set_minor_formatter(
            ticker.FixedFormatter([
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
                '13', '14', '15'
            ]))
    if classes == 9:
        ax.xaxis.set_minor_formatter(
            ticker.FixedFormatter([
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
                '13', '14', '15'
            ]))
    plt.grid(which='major')
    return bins_list

def confusion_matrix(results):
    from sklearn.metrics import confusion_matrix
    import pandas as pd
    data = confusion_matrix([item[2] for item in results], 
                          [item[1] for item in results],
                          normalize='true')
    i = max(np.unique([item[2] for item in results]).size,
               np.unique([item[1] for item in results]).size)
    df_cm = pd.DataFrame(data, columns=range(0,i),
      index=range(0,i))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize=(20, 10))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, cmap="Blues", annot=True,
              annot_kws={"size": 10})  # font size
    
def shapefile_cl(results, classes=16):
    import geopandas as gpd
    shapefile = gpd.read_file('/content/drive/My Drive/Dissertation Files/Export/Test Set New.gpkg')
    shapefile['Pred Class'] = -999
    for label in results:
        condition = shapefile['Index'] == label[0]
        shapefile.loc[condition, 'Pred Class'] = label[1]
    if classes == 16:
        bounds = [0, 2**0, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2 **\
              8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14, 2**15]
    if classes == 9:
        bounds = [0, 1, 10, 100, 500, 1000, 5000, 10000, 20000]
    for c, bound in enumerate(bounds):
        condition = shapefile['Population'] >= bound
        shapefile.loc[condition, 'Pop Class'] = int(c)
    shapefile['Class Error'] = shapefile['Pred Class'] - shapefile['Pop Class']
    shapefile['Actual Class'] = shapefile['Pop Class']
    cols = ['Index', 'Population', 'Pred Class',
            'Actual Class', 'Class Error', 'geometry']
    shapefile = shapefile[cols]
    
    fig, ax = plt.subplots(ncols=2, figsize=(30, 12))
    shapefile.plot(column='Actual Class', cmap="Purples",
                   ax=ax[0], legend=True, vmin=0, vmax=classes)
    shapefile.plot(column='Pred Class', cmap="Purples",
                   ax=ax[1], legend=True, vmin=0, vmax=classes)

    ax[0].title.set_text('Actual Population')
    ax[1].title.set_text('Model Predicted Population')

    
    fig, ax = plt.subplots(ncols=1, figsize=(15, 10))
    shapefile.plot(column='Class Error', ax=ax, cmap="coolwarm",
               legend=True, vmin=-6, vmax=6)
    plt.title("Error in Predicted Population Values")
    
    fig, ax = plt.subplots(ncols=1, figsize=(10, 10))
    sn.distplot(shapefile['Class Error'], kde=False, bins=range(0, classes))
    plt.title("Class Error Histogram")
    ax.set_xticks(range(0, classes))
    ax.set_xlabel("Magnitude of Class Error")
    ax.set_ylabel("Frequency")
    plt.show()