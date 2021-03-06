# Copyright 2019 Population Health Sciences and Image Analysis, German Center for Neurodegenerative Diseases(DZNE)
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from fatsegnet.utils.misc import create_exp_directory
import os

def get_colors(inp, colormap, vmin=None, vmax=None):
    """generate the normalize rgb values for matplolib
"""
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))


def multiview_plotting(data,labels,control_point, savepath,classes=5,alpha=0.5,nbviews=3,plot_control_point=True):
    """Plot data and label in different views
    Args:
        data: Original 3D volume
        labels: Original labels for the 3d Volume
        control_point: select the center point where the different views are going to be created
        savepath:path where the image is going to be safe
        classes: number of classes in the labeles
        alpha: transparency of the labels on the original data
        nbviews: 1 only axial view,2 axial and frontal, 3 the three views
        plot_labels: True plot labels, False only plot data


    Returns:
        An images with the diffent views and the corresponding label
"""
    # Create the colormap for the labels
    dz=np.arange(classes)
    colors = get_colors(dz, plt.cm.jet)
    #replace first color for black
    colors[0, 0:3] = [0, 0, 0]
    my_cm=LinearSegmentedColormap.from_list('mylist',colors,classes)
    plt.ioff()

    grid_size = [2, nbviews]

    #fig = plt.figure(dpi=600)
    fig, ax = plt.subplots(grid_size[0],grid_size[1])

    FIGSIZE = 4
    FIGDPI = 100
    fig.set_size_inches([FIGSIZE * grid_size[1] , FIGSIZE * grid_size[0]])
    fig.set_dpi(FIGDPI)
    fig.set_facecolor('black')
    fig.set_tight_layout({'pad': 0})
    fig.subplots_adjust(wspace=0)


    #AXIAL
    ax[0][0].imshow(data[control_point[0], :, :], cmap=cm.gray)
    if plot_control_point:
        ax[0][0].scatter(y=control_point[1], x=control_point[2], c='r', s=2)
    ax[0][0].set_axis_off()
    ax[0][0].margins(0,0)

    ax[1][0].imshow(data[control_point[0], :, :], cmap=cm.gray)
    ax[1][0].imshow(labels[control_point[0], :, :],vmin=0,vmax=classes, cmap=my_cm,alpha=alpha)
    if plot_control_point:
        ax[1][0].scatter(y=control_point[1], x=control_point[2], c='r', s=2)
    ax[1][0].set_axis_off()
    ax[1][0].margins(0, 0)

    #Coronal
    ax[0][1].imshow(data[:, control_point[1], :], cmap=cm.gray,aspect=(data.shape[1]/data.shape[0]))
    if plot_control_point:
        ax[0][1].scatter(y=control_point[0], x=control_point[2], c='r', s=2)
    ax[0][1].set_axis_off()
    ax[0][1].margins(0,0)


    ax[1][1].imshow(data[:, control_point[1], :], cmap=cm.gray,aspect=(data.shape[1]/data.shape[0]))
    ax[1][1].imshow(labels[:, control_point[1], :],vmin=0,vmax=classes, cmap=my_cm,alpha=alpha,aspect=(data.shape[1]/data.shape[0]))
    if plot_control_point:
        ax[1][1].scatter(y=control_point[0], x=control_point[2], c='r', s=2)
    ax[1][1].set_axis_off()
    ax[1][1].margins(0, 0)

    #Sagittal
    img=np.zeros((data.shape[0],data.shape[2]))
    diff_spacing=int((data.shape[2]-data.shape[1])/2)
    img[:,diff_spacing:data.shape[2]-diff_spacing]=data[:, :, control_point[2]]

    ax[0][2].imshow(img, cmap=cm.gray,aspect=(data.shape[1]/data.shape[0]))
    if plot_control_point:
        ax[0][2].scatter(y=control_point[0], x=control_point[2], c='r', s=2)
    ax[0][2].set_axis_off()
    ax[0][2].margins(0,0)

    img_label = np.zeros((data.shape[0], data.shape[2]))
    img_label[:, diff_spacing:data.shape[2] - diff_spacing] = labels[:, :, control_point[2]]
    ax[1][2].imshow(img, cmap=cm.gray, aspect=(data.shape[1] / data.shape[0]))
    ax[1][2].imshow(img_label,vmin=0,vmax=classes, cmap=my_cm,alpha=alpha, aspect=(data.shape[1] / data.shape[0]))
    if plot_control_point:
        ax[1][2].scatter(y=control_point[0], x=control_point[2], c='r', s=2)
    ax[1][2].set_axis_off()
    ax[1][2].margins(0, 0)


    plt.savefig(savepath, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_predictions(fat_array,pred_array,save_path):

        # Modified images for display
        disp_fat = np.flipud(np.swapaxes(fat_array[:], 0, 2))
        disp_fat = np.fliplr(disp_fat[:])
        disp_pred = np.copy(np.swapaxes(pred_array, 0, 2))
        disp_pred = np.flipud(disp_pred)
        disp_pred = np.fliplr(disp_pred)

        # only display SAT and VAT
        disp_pred[disp_pred >= 3] = 0

        idx = (np.where(disp_pred > 0))
        low_idx = np.min(idx[0])
        high_idx = np.max(idx[0])

        interval = (high_idx - low_idx) // 4

        # Control images of the segmentation

        create_exp_directory(os.path.join(save_path, 'QC'))
        for i in range(4):
            control_point = [0, int(np.ceil(disp_fat.shape[1] / 2)), int(np.ceil(disp_fat.shape[2] / 2))]
            control_point[0] = int(np.ceil(np.random.uniform(high_idx - interval * i, high_idx - interval * ((i + 1)))))
            multiview_plotting(disp_fat, disp_pred, control_point, save_path + '/QC/QC_%s.png' % i,
                               classes=5, alpha=0.5, nbviews=3)