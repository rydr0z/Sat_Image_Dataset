import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import torch
import matplotlib.pyplot as plt
import geopandas as gpd


def population_hist(dataset, bins=10, figsize=(20, 10)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
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
    ax.xaxis.set_minor_locator(
        ticker.FixedLocator(np.arange(0, classes+1) + 0.5))

    if classes == 16:
        ax.xaxis.set_minor_formatter(
            ticker.FixedFormatter([
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
                '13', '14', '15'
            ]))

    if classes == 6:
        ax.xaxis.set_minor_formatter(
            ticker.FixedFormatter([
                '0', '1', '2', '3', '4', '5'
            ]))
    return bins_list


def population_hist(dataset, bins=10, figsize=(20, 10)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    pop_list = []
    for i in range(0, len(dataset)):
        pop = dataset[i][1][1].item()
        pop_list.append(pop)
    bins_list = sn.distplot(pop_list, bins=bins, kde=False, norm_hist=False)
    return bins_list


def balanced_class_hist(train_loader, figsize=(20, 10), classes=16):
    import matplotlib.ticker as ticker
    bins = range(0, classes+1)
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
    ax.xaxis.set_minor_locator(
        ticker.FixedLocator(np.arange(0, classes+1) + 0.5))
    if classes == 16:
        ax.xaxis.set_minor_formatter(
            ticker.FixedFormatter([
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
                '13', '14', '15'
            ]))
    if classes == 6:
        ax.xaxis.set_minor_formatter(
            ticker.FixedFormatter([
                '0', '1', '2', '3', '4', '5'
            ]))
    plt.grid(which='major')
    return bins_list


def confusion_matrix(results):
	from sklearn.metrics import confusion_matrix
    data = confusion_matrix([item[2] for item in results],
                            [item[1] for item in results],
                            normalize=None)
    i = max(np.unique([item[2] for item in results]).size,
            np.unique([item[1] for item in results]).size)
    df_cm = pd.DataFrame(data, columns=range(0, i),
                         index=range(0, i))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize=(20, 10))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, cmap="Blues", annot=True,
               annot_kws={"size": 10})  # font size


def shapefile_cl(results, classes=16):
    shapefile = gpd.read_file(
        '/content/drive/My Drive/Dissertation Files/Export/Test Set New.gpkg')
    shapefile['Pred Class'] = -999
    for label in results:
        condition = shapefile['Index'] == label[0]
        shapefile.loc[condition, 'Pred Class'] = label[1]
    if classes == 16:
        bounds = [0, 2**0, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2 **
                  8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14, 2**15]
    if classes == 6:
        bounds = [0, 1, 10, 100, 1000, 10000]
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