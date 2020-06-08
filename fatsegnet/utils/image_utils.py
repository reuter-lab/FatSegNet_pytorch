
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


import numpy as np
from skimage.measure import label

def find_labels(arr,axis=-1):
    idx=(np.where(arr > 0))
    min_idx=np.min(idx[axis])
    max_idx=np.max(idx[axis])
    return max_idx,min_idx

def find_unique_index_slice(data):

    aux_index=[]

    for z in range(data.shape[-1]):
        labels,counts=np.unique(data[:,:,z],return_counts=True)
        if 2 in labels:
            num_pixels=np.sum(counts[1:])
            position=np.where(labels==2)
            if counts[position[0][0]] >= (num_pixels*0.8):
                aux_index.append(z)


    higher_index=np.max(aux_index)
    lower_index= np.min(aux_index)

    return higher_index,lower_index


def plane_swap(data,plane,inverse=False):
    if plane == 'axial':
        if not inverse:
            return np.moveaxis(data,[0,1,2],[2,1,0])
        else:
            return np.moveaxis(data,[0,1,2],[2,1,0])

    elif plane == 'coronal':
        if not inverse:
            return np.moveaxis(data,[0,1,2],[2,0,1])
        else:
            return np.moveaxis(data,[0, 1, 2],[1,2,0])
    elif plane == 'sagittal':
        return data


def define_size(mov_dim,ref_dim):
    """Calculate a new image size by duplicate the size of the bigger ones
    Args:
        move_dim (3D array sise):  3D size of the input volume
        ref_dim (3D ref size) : 3D size of the reference size
    Returns:
        new_dim (list) : New array size
        borders (list) : border Index for mapping the old volume into the new one
    """
    new_dim=np.zeros(len(mov_dim),dtype=np.int)
    borders=np.zeros((len(mov_dim),2),dtype=int)

    padd = [int(mov_dim[0] // 2), int(mov_dim[1] // 2), int(mov_dim[2] // 2)]

    for i in range(len(mov_dim)):
        new_dim[i]=int(max(2*mov_dim[i],2*ref_dim[i]))
        borders[i,0]= int(new_dim[i] // 2) -padd [i]
        borders[i,1]= borders[i,0] +mov_dim[i]

    return list(new_dim),borders

def map_size(arr,base_shape,verbose=1):
    """Padd or crop the size of an input volume to a reference shape
    Args:
        arr (3D array array):  array to be map
        base_shape (3D ref size) : 3D size of the reference size
    Returns:
        final_arr (3D array) : 3D array containing with a shape defined by base_shape
    """
    if verbose >0:
        print('Volume will be resize from %s to %s ' % (arr.shape, base_shape))

    new_shape,borders=define_size(np.array(arr.shape),np.array(base_shape))
    new_arr=np.zeros(new_shape)
    final_arr=np.zeros(base_shape)

    new_arr[borders[0,0]:borders[0,1],borders[1,0]:borders[1,1],borders[2,0]:borders[2,1]]= arr[:]

    middle_point = [int(new_arr.shape[0] // 2), int(new_arr.shape[1] // 2), int(new_arr.shape[2] // 2)]
    padd = [int(base_shape[0]/2), int(base_shape[1]/2), int(base_shape[2]/2)]

    low_border=np.array((np.array(middle_point)-np.array(padd)),dtype=int)
    high_border=np.array(np.array(low_border)+np.array(base_shape),dtype=int)

    final_arr[:,:,:]= new_arr[low_border[0]:high_border[0],
                   low_border[1]:high_border[1],
                   low_border[2]:high_border[2]]

    return final_arr


def get_largest_cc(segmentation):
    """
    Function to find largest connected component of segmentation.
    :param np.ndarray segmentation: segmentation
    :return:
    """
    labels = label(segmentation, connectivity=3, background=0)

    bincount = np.bincount(labels.flat)

    value = np.argmax(bincount)

    if value == 0:
        background = np.argmax(bincount)
        bincount[background] = -1
        value= np.argmax(bincount)

    largest_cc = labels == value

    return largest_cc

def remove_small_regions(segmentation,tolerance=10):

    labels= label(segmentation, background=0)
    bincount = np.bincount(labels.flat)

    for i in range(len(bincount)):
        if bincount[i] <= tolerance :
            segmentation[labels == i] = False

    return segmentation