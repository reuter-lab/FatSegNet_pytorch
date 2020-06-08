
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

import argparse
import sys
import os
import time

sys.path.append('../')

from fatsegnet.utils.misc import locate_file, locate_dir ,create_exp_directory ,setup_logger
import pandas as pd
import numpy as np


def check_paths(args,subject_id):

    #output_dir
    create_exp_directory(args.output_dir)

    #sub_dir
    final_path = os.path.join(args.output_dir,subject_id)
    create_exp_directory(final_path)

    return final_path


def run_adipose_pipeline(args,flags,save_path='/',data_path='/',id='Test',logger='log.txt'):
    from fatsegnet.utils.conform import conform
    from fatsegnet.model.FatSegNet import FatSegNet
    from fatsegnet.utils.metrics import calculate_statistics, stats_variable_initialization
    from fatsegnet.utils.visualization_misc import plot_predictions
    import nibabel as nib

    output_stats = 'AAT_stats.tsv'
    output_pred_fat = 'AAT_pred.nii.gz'
    qc_images = []

    logger.info('-' * 30)
    logger.info('Loading Subject')
    logger.info(id)
    sub = id

    fat_file = locate_file('*'+str(args.fat_image), data_path)
    water_file = locate_file('*'+str(args.water_image), data_path)

    # Check fat
    if fat_file:
        logger.info('-' * 30)
        logger.info('Loading Fat Image')
        logger.info(fat_file[0])
        #Load Fat Images
        fat_img = nib.load(fat_file[0])
        ishape = fat_img.shape

        #Check if  data from example_data_folder was loaded : Only contains the value -9999
        if len(np.unique(fat_img.get_data())) > 2:
            if len(ishape) > 3 and ishape[3] != 1:
                logger.info('ERROR: Multiple input frames (' + format(fat_img.shape[3]) + ') not supported!')
            else:
                fat_img = conform(fat_img, flags=flags, order=args.order, save_path=save_path, mod='fat',
                                  axial=args.axial,logger=logger)
                fat_array = fat_img.get_data()
                fat_zooms = fat_img.header.get_zooms()

            logger.info('-' * 30)
            logger.info('Loading Water Image')
            #Check water image
            if not water_file:
                weighted=False
                logger.info('No water image found, weighted volumes would not be calculated')
                water_array=np.zeros(fat_array.shape)
            else:
                logger.info(water_file[0])
                weighted=True
                water_img = nib.load(water_file[0])
                ishape = fat_img.shape
                if len(ishape) > 3 and ishape[3] != 1:
                    logger.info('ERROR: Multiple input frames (' + format(water_img.shape[3]) + ') not supported!')
                    weighted = False
                    logger.info('No water image found, weighted volumes would not be calculated')
                    water_array = np.zeros(fat_array.shape)
                else:
                    water_img = conform(water_img, flags=flags, order=args.order, save_path=save_path, mod='water',
                                       axial=args.axial,logger=logger)
                    water_array = water_img.get_data()


            variable_columns, base_variable_len = stats_variable_initialization(weighted)

            pixel_matrix = np.zeros((1, len(variable_columns)), dtype=object)

            img_spacing=np.copy(fat_zooms)

            pipeline= FatSegNet(flags=flags,args=args,logger=logger)

            pred_array, high_idx, low_idx = pipeline.eval(fat_array)


            logger.info('-' * 30)
            logger.info('Calculating Stats')

            pixel_matrix[0, 0] = sub
            pixel_matrix[0, 1] = int(high_idx - low_idx)

            pixel_matrix[0, 2:] = calculate_statistics(pred_array[ :, :,low_idx:high_idx],
                                                                          water_array[:, :,low_idx:high_idx],
                                                                          fat_array[:, :,low_idx:high_idx],
                                                                          low_idx, high_idx, variable_columns[2:],
                                                                          base_variable_len, img_spacing, weighted=weighted)



            df = pd.DataFrame(pixel_matrix, columns=variable_columns)

            seg_path=os.path.join(save_path, 'Segmentations')

            create_exp_directory(seg_path)

            df.to_csv(seg_path+'/'+output_stats, sep='\t', index=False)
            df.to_json(seg_path+ '/AAT_variables_summary.json', orient='records')


            if not args.control_images:
                plot_predictions(fat_array,pred_array,save_path)

            logger.info('-' * 30)
            logger.info('Saving Segmentation')
            # Save prediction
            pred_img = nib.Nifti1Image(pred_array, fat_img.affine, fat_img.header)
            nib.save(pred_img, seg_path+'/'+output_pred_fat)

            logger.info('-' * 30)

            logger.info('Finish Subject %s' % sub)

            logger.info('-' * 30)


        else :
            logger.info('ERROR: Input image empty \n'
                  'Note : Volumes from the example_data_folder are empty \n'
                  'The example_data_folder is only a ilustrative example on how volumes have to be organized for FatSegNet to work.')
            logger.info('Please provided your own dixon MR scans')

    else:
        logger.info('')
        logger.info('-' * 30)
        logger.info('ERROR: Subject doesnt have a Fat Image named %s,\n'
              'Please verified that the name provided to the -fat argument matches the one in the participants folder (default : FatImaging_F.nii.gz )'%str(args.fat_image))


def option_parse():
    parser = argparse.ArgumentParser(
        description='FatSegNet pipeline for segmentation of abdominal adipose tissue into VAT and SAT and localization of the abdominal region',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument("-in_dir", "--data_dir", help="Data directory", required=True)

    parser.add_argument("-out_dir", "--output_dir", help="Main output directory where models results are going to be store", required=True)

    parser.add_argument("-f", "--file", help="One compulsory column csv file containing the subjects to process", required=False,default='participants.csv')

    parser.add_argument("-fat", "--fat_image", type=str, help="Name of the fat image", required=False,
                        default='FatImaging_F.nii.gz')
    parser.add_argument("-water", "--water_image", type=str, help="Name of the water image", required=False,
                        default='FatImaging_W.nii.gz')

    parser.add_argument('-loc', "--run_localization", action='store_true',
                        help='Run abdominal region localization model ', required=False)

    parser.add_argument('-axial', "--axial", action='store_true',
                        help='Run only axial segmentation model ', required=False, default=False)
    parser.add_argument('-no_qc',"--control_images",action='store_true',help='Disable prediction plots for quality control',required=False)


    parser.add_argument('-order', "--order", type=int,
                        help='Interpolation order (0=nearest,1=linear(default),2=quadratic,3=cubic), only use if input image have to be standarized ', required=False, default=1)

    parser.add_argument('-batch', "--batch_size", type=int,
                        help='Batch size for inference by default is 16', required=False, default=16)

    parser.add_argument('-clean', "--cleanup", action='store_true',
                        help='Clean final segmentation', required=False)

    parser.add_argument('-gpu_id', "--gpu_id", type=int,
                        help='GPU device name to run model', required=False, default=0)
    parser.add_argument('-no_cuda', "--no_cuda", action='store_true',
                        help='Disable CUDA (no GPU usage, inference on CPU)', required=False)



    args = parser.parse_args()


    FLAGS = {}
    # Segmenation model
    FLAGS['segmentation'] = {}

    FLAGS['segmentation'][
        'axial'] = '../fatsegnet/checkpoints/segmentation/CDFNet_axial/ckpts/Epoch_40_training_state.pkl'
    FLAGS['segmentation'][
        'coronal'] = '../fatsegnet/checkpoints/segmentation/CDFNet_coronal/ckpts/Epoch_40_training_state.pkl'
    FLAGS['segmentation'][
        'sagittal'] = '../fatsegnet/checkpoints/segmentation/CDFNet_sagittal/ckpts/Epoch_40_training_state.pkl'

    FLAGS['localization'] = {}
    FLAGS['localization'][
        'coronal'] = '../fatsegnet/checkpoints/localization/CDFNet_coronal/ckpts/Epoch_30_training_state.pkl'
    FLAGS['localization'][
        'sagittal'] = '../fatsegnet/checkpoints/localization/CDFNet_sagittal/ckpts/Epoch_30_training_state.pkl'

    FLAGS['imgSize'] = [256, 224, 72]
    FLAGS['spacing'] = [float(1.9531), float(1.9531), float(5.0)]
    FLAGS['base_ornt'] = np.array([[0, -1], [1, 1], [2, 1]])



    return args,FLAGS


def run_fatsegnet(args,FLAGS):

    # load file
    participant_file=locate_file('*'+args.file,args.data_dir)
    if participant_file:
        print('Loading participant from file : %s'%participant_file[0])
        df =pd.read_csv(participant_file[0],header=None)
        if df.empty:
            print('Participant file empty ')
        else:
            file_list=df.values
            for sub in file_list:
                id=sub[0]
                path = locate_dir('*'+str(id)+'*',args.data_dir)
                if path:
                    if os.path.isdir(path[0]):

                        start = time.time()

                        save_path = check_paths(args=args, subject_id=str(id))

                        logger = setup_logger(os.path.join(save_path, "log.txt"))

                        run_adipose_pipeline(args=args, flags=FLAGS, save_path=save_path,data_path=path[0],id=str(id),logger=logger)

                        end = time.time() - start

                        logger.info("Total computation time :  %0.4f seconds."%end)

                    else:
                        print ('Directory %s not found'%path)
                else :
                    print('Directory name %s not found' % id)

            print('\n')
            print('Thank you for using FatSegNet')
            print('If you find it useful and use it for a publication, please cite: ')
            print('\n')
            print('Estrada S, Lu R, Conjeti S, et al.'
                  'FatSegNet: A fully automated deep learning pipeline for adipose tissue segmentation on abdominal dixon MRI.'
                  'Magn Reson Med. 2019;00:1-13. https:// doi.org/10.1002/mrm.28022')
    else:
        print('No partipant file found, please provide one the input data folder')



if __name__=='__main__':


    args,FLAGS= option_parse()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
    # The GPU id to use, usually either "0" or "1";
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id);

    run_fatsegnet(args,FLAGS)

    sys.exit(0)

