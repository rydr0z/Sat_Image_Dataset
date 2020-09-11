import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import torch
import matplotlib.pyplot as plt
import geopandas as gpd

from matplotlib import rc
import matplotlib.pylab as plt

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


def fonts(small, medium, large):
    """function for setting font sizes as desired"""
    SMALL_SIZE = small
    MEDIUM_SIZE = medium
    BIGGER_SIZE = large
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE,
           titleweight='bold')  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE,
           titleweight='bold')  # fontsize of the figure title


def large_fonts():
    """function for setting large font sizes"""
    SMALL_SIZE = 20
    MEDIUM_SIZE = 30
    BIGGER_SIZE = 40
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE,
           titleweight='bold')  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE,
           titleweight='bold')  # fontsize of the figure title


def small_fonts():
    """function for setting small font sizes"""
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 22

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# Use large fonts by default
large_fonts()


def population_hist(dataset, bins=10, figsize=(25, 10)):
    """Creates a population hist for dataset"""
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    pop_list = []
    for i in range(0, len(dataset)):
        pop = dataset[i][1][1].item()
        pop_list.append(pop)
    bins_list = axs[0].hist(pop_list, bins=bins)
    axs[0].set_ylabel('Frequency')
    axs[1].hist(pop_list, bins=bins)
    axs[1].set_yscale('log')
    axs[1].set_ylabel('Frequency (Log-Scaled)')
    for ax in axs:
        ax.set_xlabel('Population')
    fig.suptitle('Population Histogram', fontsize=50, fontweight='bold')
    return bins_list


def class_hist(dataset, figsize=(10, 10), classes=16):
    """Creates a class hist for dataset"""
    import matplotlib.ticker as ticker
    bins = range(0, classes + 1)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    class_list = []
    for i in range(0, len(dataset)):
        cl = dataset[i][1][2].item()
        class_list.append(cl)
    bins_list = plt.hist(class_list, bins=bins, edgecolor='black', linewidth=1)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.xaxis.set_minor_locator(
        ticker.FixedLocator(np.arange(0, classes + 1) + 0.5))

    if classes == 16:
        ax.xaxis.set_minor_formatter(
            ticker.FixedFormatter([
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
                '12', '13', '14', '15'
            ]))

    if classes == 6:
        ax.xaxis.set_minor_formatter(
            ticker.FixedFormatter(['0', '1', '2', '3', '4', '5']))
    return bins_list


def balanced_class_hist(train_loader, figsize=(20, 10), classes=16):
    """Creates a balanced class hist for dataset"""
    import matplotlib.ticker as ticker
    bins = range(0, classes + 1)
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
        ticker.FixedLocator(np.arange(0, classes + 1) + 0.5))
    if classes == 16:
        ax.xaxis.set_minor_formatter(
            ticker.FixedFormatter([
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
                '12', '13', '14', '15'
            ]))
    if classes == 6:
        ax.xaxis.set_minor_formatter(
            ticker.FixedFormatter(['0', '1', '2', '3', '4', '5']))
    plt.grid(which='major')
    return bins_list


def confusion_matrix(results):
    """Creates a confusion matrix based on produced results"""
    from sklearn.metrics import confusion_matrix
    data = confusion_matrix([item[2] for item in results],
                            [item[1] for item in results],
                            normalize='true')
    i = max(
        np.unique([item[2] for item in results]).size,
        np.unique([item[1] for item in results]).size)
    df_cm = pd.DataFrame(data, columns=range(0, i), index=range(0, i))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize=(20, 10))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 10})


def shapefile_cl(results, classes=16, test_region=1):
    """Matches predictions from results with their position in shapefile
    and produces classification visualisations."""
    if test_region == 1:
        shapefile = gpd.read_file(
            '/content/drive/My Drive/Dissertation Files/Export/Test Set New.gpkg'
        )
        fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(32, 8))
    else:
        shapefile = gpd.read_file(
            '/content/drive/My Drive/Dissertation Files/Export/Test Set 1.gpkg'
        )
        fig, ax = plt.subplots(ncols=1, nrows=3, figsize=(30, 15))
    shapefile['Pred Class'] = -999
    for label in results:
        condition = shapefile['Index'] == label[0]
        shapefile.loc[condition, 'Pred Class'] = label[1]
    if classes == 16:
        bounds = [
            0, 2**0, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9,
            2**10, 2**11, 2**12, 2**13, 2**14, 2**15
        ]
    if classes == 6:
        bounds = [0, 1, 10, 100, 1000, 10000]
    for c, bound in enumerate(bounds):
        condition = shapefile['Population'] >= bound
        shapefile.loc[condition, 'Pop Class'] = int(c)
    shapefile['Class Error'] = shapefile['Pred Class'] - shapefile['Pop Class']
    shapefile['Actual Class'] = shapefile['Pop Class']
    cols = [
        'Index', 'Population', 'Pred Class', 'Actual Class', 'Class Error',
        'geometry'
    ]
    shapefile = shapefile[cols]

    vmin = -10
    vmax = 10
    if classes == 6:
        vmin = -5
        vmax = 5
    shapefile.plot(column='Actual Class',
                   cmap="Reds",
                   ax=ax[0],
                   legend=True,
                   vmin=0,
                   vmax=classes)
    ax[0].title.set_text('Actual Population')
    shapefile.plot(column='Pred Class',
                   cmap="Reds",
                   ax=ax[1],
                   legend=True,
                   vmin=0,
                   vmax=classes)
    ax[1].title.set_text('Model Predicted Population')
    shapefile.plot(column='Class Error',
                   ax=ax[2],
                   cmap="coolwarm",
                   legend=True,
                   vmin=vmin,
                   vmax=vmax)
    ax[2].title.set_text("Prediction Error")

    fig, ax = plt.subplots(ncols=1, figsize=(15, 10))
    sn.distplot(shapefile['Class Error'], kde=False, bins=range(0, classes))
    plt.title("Class Error Histogram")
    ax.set_xticks(range(0, classes))
    ax.set_xlabel("Magnitude of Class Error")
    ax.set_ylabel("Frequency")
    plt.show()
    return shapefile


def shapefile_reg(results):
    """Matches predictions from results with their position in shapefile
    and produces regression visualisations."""
    shapefile = gpd.read_file(
        '/content/drive/My Drive/Dissertation Files/Export/Test Set New.gpkg')
    shapefile['Pred Population'] = -999
    for label in results:
        condition = shapefile['Index'] == label[0]
        shapefile.loc[condition, 'Pred Population'] = label[1]
    shapefile['Population Error'] = shapefile['Pred Population'] - shapefile[
        'Population']
    cols = [
        'Index', 'Population', 'Pred Population', 'Population Error',
        'geometry'
    ]
    shapefile = shapefile[cols]

    fig, ax = plt.subplots(ncols=3, figsize=(32, 8))
    shapefile.plot(column='Population',
                   cmap="Purples",
                   ax=ax[0],
                   legend=True,
                   vmin=0,
                   vmax=shapefile['Population'].max())
    shapefile.plot(column='Pred Population',
                   cmap="Purples",
                   ax=ax[1],
                   legend=True,
                   vmin=0,
                   vmax=shapefile['Population'].max())
    ax[0].title.set_text('Actual Population')
    ax[1].title.set_text('Model Predicted Population')
    shapefile.plot(column='Population Error',
                   ax=ax[2],
                   cmap="coolwarm",
                   legend=True,
                   vmin=-2000,
                   vmax=2000)
    ax[2].title.set_text("Error in Predicted Population Values")

    fig, ax = plt.subplots(ncols=1, figsize=(16, 8))
    sn.distplot(shapefile['Population Error'], kde=False)
    plt.title("Population Error Error Histogram")
    ax.set_xlabel("Population Error")
    ax.set_ylabel("Frequency")
    ax.set_yscale("Log")
    plt.show()
    return shapefile


def display_layer_activations(model, image_label_pair, device, sat_mean,
                              sat_std):
    """Displays all layer activations for a given image and label passed through model"""

    from skimage import data, img_as_float
    from skimage import exposure

    net = model
    net.to('cpu')

    visualisation = {}

    def hook_fn(m, i, o):
        visualisation[m] = o

    def get_all_layers(net):
        for name, layer in net._modules.items():
            layer.to(device)
            #If it is a sequential, don't register a hook on it
            # but recursively register hook on all it's module children
            if isinstance(layer, torch.nn.Sequential):
                get_all_layers(layer)
            else:
                # it's a non sequential. Register a hook
                layer.register_forward_hook(hook_fn)

    get_all_layers(net)

    images, labels = image_label_pair
    images, labels = images.to('cpu'), labels.to('cpu')

    vis_image = (images * sat_std) + sat_mean
    band_cmaps = [
        'Blues', "Greens", "Reds", 'YlOrRd', 'YlOrRd', 'Greys', 'YlOrRd'
    ]
    fig, axs = plt.subplots(1, 8, figsize=(20, 10))
    for i, ax in enumerate(axs.ravel()):
        if i == 0:
            im = vis_image[0][0:3].permute(2, 1, 0) / 3000
            ax.imshow(im)
            ax.set_title("RGB Bands")
        else:
            ax.imshow(vis_image[0].permute(2, 1, 0)[:, :, i - 1],
                      cmap=band_cmaps[i - 1])
            ax.set_title("Band {}".format(i))
    fig.suptitle('Population: {}, Class: {}'.format(labels[0][1].item(),
                                                    labels[0][2].item()),
                 y=0.7)

    images = images.float()

    out = net.float().to('cpu')(images)

    # Just to check whether we got all layers
    print("Captured forward hooks for image."
          )  #output includes sequential layers

    from torchvision import utils
    keys_list = list(visualisation.keys())
    for layer in range(0, len(keys_list)):
        print('Layer {}: {}'.format(layer, keys_list[layer]))
        activation = visualisation[keys_list[layer]]
        activation = activation.permute(1, 0, 2, 3)
        
        nrow = 8
        grid = utils.make_grid(activation,
                               nrow=nrow,
                               normalize=True,
                               padding=0)
        grid = grid.detach().numpy()

        fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))

        ax1.imshow(grid.transpose(2, 1, 0),
                  interpolation='none',
                  cmap='viridis')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_xticks([])
        ax1.set_yticks([])
        layer_name = str(keys_list[layer]).split('(')[0]

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.ioff()
    plt.show()


def show_images(band, cmap, dataset, seed, sat_mean, sat_std):
    """Shows 5 clipped satellite images.
    Arguments:
        band (int or list(int)): the bands to be displayed selected from 0,1,2,3,4,5,6,7
        cmap (string): the colormap for the plot
        dataset: SatImageDataset containing images
        seed (int): select seed to display same images each time function executed
    """

    from skimage import data, img_as_float
    from skimage import exposure

    torch.manual_seed(seed)
    train_loader_for_vis = torch.utils.data.DataLoader(dataset,
                                                       batch_size=1,
                                                       shuffle=True)

    vis_images = []
    vis_labels = []

    data_iter = iter(train_loader_for_vis)
    for i in range(1, 6):
        vis_image, vis_label = data_iter.next()
        vis_images.append(vis_image)
        vis_labels.append(vis_label)

    fig = plt.figure(figsize=(33, 50))
    columns = 5
    rows = 1

    ax = []
    plt.rcParams.update({'font.size': 18})
    for i in range(1, columns * rows + 1):
        image, label = vis_images[i - 1], vis_labels[i - 1]
        image = (image * sat_std) + sat_mean
        ax.append(fig.add_subplot(rows, columns, i))
        ax[-1].set_title("Image {i}\nPopulation: {p}\nClass: {c}".format(
            i=i, p=str(label[:, 1, :].item()), c=str(label[:, 2, :].item())))
        if band != 5:
            plt.imshow(image[0].permute(2, 1, 0)[:, :, band],
                       cmap=cmap,
                       vmin=0,
                       vmax=1)
        if band == (0, 1, 2):
            im = image[0][0:3].permute(2, 1, 0) / 3000
            im = im.numpy()
            im = img_as_float(im)
            im = exposure.adjust_gamma(im, 0.9)
            plt.imshow(im)
        else:
            plt.imshow(image[0].permute(2, 1, 0)[:, :, band], cmap=cmap)
    plt.show()
    plt.rcParams.update({'font.size': 12})
    plt.savefig(
        "\content\drive\My Drive\Dissertation Files\Exported Figures\Example Images - Band {}"
        .format(band),
        format="png")