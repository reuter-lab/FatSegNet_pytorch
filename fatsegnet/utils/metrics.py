
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
from skimage.measure import perimeter

def stats_variable_initialization(weighted=True):

    # initialize Stats Variables
    variable_columns = []

    volume_variable_columns = ['VOL_cm3', 'SAT_VOL_cm3', 'VAT_VOL_cm3', 'AAT_VOL_cm3',
                               'VAT_VOL_TO_SAT_VOL', 'VAT_VOL_TO_AAT_VOL', 'SAT_VOL_TO_AAT_VOL']

    w_volume_variable_columns= ['W_VOL_cm3','WSAT_VOL_cm3', 'WVAT_VOL_cm3',
                               'WAAT_VOL_cm3', 'WVAT_VOL_TO_WSAT_VOL', 'WVAT_VOL_TO_WAAT_VOL', 'WSAT_VOL_TO_WAAT_VOL']

    area_variable_columns = ['HEIGHT_cm', 'AVG_AREA_cm2', 'AVG_PERIMETER_cm']

    base_variable_len={}
    base_variable_len['Area']=len(area_variable_columns)
    base_variable_len['Volume']=len(volume_variable_columns)
    base_variable_len['W_Volume']=len(w_volume_variable_columns)

    roi_areas = ['wb']

    for roi in roi_areas:
        for area_id in area_variable_columns:
            variable_columns.append(roi + '_' + area_id)
        for vol_id in volume_variable_columns:
            variable_columns.append(roi + '_' + vol_id)

        if weighted:
            for w_vol_id in w_volume_variable_columns:
                variable_columns.append(roi + '_' + w_vol_id)


    variable_columns.insert(0, 'imageid')
    variable_columns.insert(1, 'Slices_ROI')

    return variable_columns,base_variable_len

def perimeter_calculation(label_mask):

    perimeter_val=[]

    for slice in range(label_mask.shape[-1]):
        perimeter_val.append(perimeter(label_mask[:,:,slice]))

    average_perimeter = np.sum(perimeter_val)/label_mask.shape[-1]

    return  average_perimeter

def calculate_areas(final_img, img_spacing,columns):


    pixel_area = (img_spacing[0] * img_spacing[1]) * 0.01

    statiscs_matrix = np.zeros(( 1, columns),dtype=np.float64)

    abdominal_region_mask= np.zeros(final_img.shape,dtype=bool)
    abdominal_region_mask[final_img >= 1] = True

    #  Metric Measurements
    statiscs_matrix[0,0]= final_img.shape[-1] * img_spacing[2] * 0.1 # Height ROI
    statiscs_matrix[0,1]= (np.sum(abdominal_region_mask) * pixel_area) / final_img.shape[-1] #Average_Area
    statiscs_matrix[0, 2] = perimeter_calculation(abdominal_region_mask) * img_spacing[0] * 0.1  # Average_perimeter

    return  statiscs_matrix.round(decimals=4)

def calculate_volumes(final_img, water_array, fat_array, img_spacing,columns,weighted=True):

    voxel_volume = (img_spacing[0] * img_spacing[1] * img_spacing[2]) * 0.001

    abdominal_region_mask= np.zeros(final_img.shape,dtype=bool)
    abdominal_region_mask[final_img >= 1] = True

    vat_mask = np.zeros(final_img.shape, dtype=bool)
    vat_mask[final_img == 2] = True

    sat_mask = np.zeros(final_img.shape, dtype=bool)
    sat_mask[final_img == 1] = True

    combine_array = water_array + fat_array
    fat_fraction_array = fat_array[:] / np.clip(combine_array,0.00001,None)

    if weighted:
        vat_fraction = np.sum(fat_fraction_array[vat_mask])
        sat_fraction = np.sum(fat_fraction_array[sat_mask])
        abdominal_region_fraction= np.sum (fat_fraction_array[abdominal_region_mask])
    else:
        sat_fraction = np.sum(sat_mask)
        vat_fraction = np.sum(vat_mask)
        abdominal_region_fraction= np.sum (abdominal_region_mask)

    #print('the vat fraction values are %d, the sat fraction values are %d' % (vat_fraction, sat_fraction))

    statiscs_matrix = np.zeros(( 1, columns),dtype=np.float64)

    #  Metric Measurements
    statiscs_matrix[0,0] = abdominal_region_fraction * voxel_volume # Volume of Abdominal Region

    # Pixel not Weighted
    statiscs_matrix[0, 1] = sat_fraction * voxel_volume  # VOL_SAT
    statiscs_matrix[0, 2] = vat_fraction * voxel_volume  # VOL_VAT
    statiscs_matrix[0, 3] = statiscs_matrix[0, 1] + statiscs_matrix[0, 2]  # VOL_AAT

    statiscs_matrix[0, 4] = statiscs_matrix[0,2] / statiscs_matrix[0, 1]  # VAT/SAT
    statiscs_matrix[0, 5] = statiscs_matrix[0, 2] / statiscs_matrix[0, 3]  # VAT/AAT
    statiscs_matrix[0, 6] = statiscs_matrix[0, 1] / statiscs_matrix[0, 3]  # SAT/AAT

    #statiscs_matrix[0,17]=extreme_AAT_increase_flag(final_img,threshold=increase_thr)

    return statiscs_matrix.round(decimals=4)

def calculate_statistics(final_img, water_array, fat_array, low_idx, high_idx, columns,base_variables_len,img_spacing,weighted=True):
    """
    Rhineland Stuty version
    :param final_img:
    :param water_array:
    :param fat_array:
    :param low_idx:
    :param high_idx:
    :param columns:
    :param base_variables_len:
    :param img_spacing:
    :param increase_thr:
    :param comparments:
    :param weighted:
    :return:
    """


    statiscs_matrix = np.zeros((1, len(columns)),dtype=object)

    #print('Whole Body')

    area_idx=base_variables_len['Area']
    statiscs_matrix[0, 0:area_idx]=calculate_areas(final_img,img_spacing,base_variables_len['Area'])
    volume_idx=area_idx +base_variables_len['Volume']
    statiscs_matrix[0,area_idx:volume_idx]=calculate_volumes(final_img,water_array,fat_array,
                                                                              img_spacing, base_variables_len['Volume'], weighted=False)

    if weighted:
        statiscs_matrix[0,volume_idx:] = calculate_volumes(final_img,water_array,fat_array,img_spacing,base_variables_len['Volume'],weighted=True)
    return statiscs_matrix



