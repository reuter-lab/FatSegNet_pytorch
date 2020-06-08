
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


import torch
import numpy as np
import time
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from fatsegnet.utils.datasetUtils import testDataset
from fatsegnet.utils.transformUtils import ToTensorTest, ToTensorTest_logits
from fatsegnet.utils.image_utils import get_largest_cc, plane_swap, map_size ,find_unique_index_slice,remove_small_regions
from fatsegnet.model.CDFNet import CDFNet
from fatsegnet.model.view_agg import view_agg


class FatSegNet(object):

    params_network = {'num_channels': 1, 'num_filters': 64,
                      'kernel_h': 5, 'kernel_w': 5, 'stride_conv': 1,
                      'pool': 2, 'stride_pool': 2, 'num_classes': 5,
                      'kernel_c': 1, 'kernel_d': 1, 'batch_size': 8,
                      'height': 256, 'width': 256}

    view_agg_network = {'num_channels': 5, 'num_filters': 30,
                      'kernel_h': 5, 'kernel_w': 5 ,'kernel_d' :5,
                      'stride_conv': 1, 'num_classes': 5,'depth':72 ,'height':224 ,'width': 256}

    def __init__(self, flags, args,logger):

        self.flags = flags
        self.params_network = FatSegNet.params_network.copy()
        self.view_agg_network = FatSegNet.view_agg_network.copy()
        self.args = args
        self.logger = logger
        self.device,self.use_cuda=self.check_device()

    def check_device(self):
        # Put it onto the GPU or CPU
        use_cuda = not self.args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.logger.info("Cuda available: {}, "
              "# Available GPUS: {}, "
              "Cuda user disabled (--no_cuda flag): {}, "
              "--> Using device: {}".format(torch.cuda.is_available(), torch.cuda.device_count(), self.args.no_cuda, device))
        return device,use_cuda


    def predict(self,img,batch_size,model,plane,orig_shape):
        transform_test = transforms.Compose([ToTensorTest()])
        test_dataset = testDataset(img, transforms=transform_test)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

        pred_logits = []

        model.eval()
        with torch.no_grad():
            for batch_idx, sample_batch in enumerate(test_loader):
                images_batch = sample_batch['image']

                if self.use_cuda:
                    images_batch = images_batch.cuda()

                temp = model(images_batch)
                pred_logits.append(temp.cpu())

            pred_logits = torch.cat(pred_logits, axis=0)

        # change from N,C,W,H to view with C in last dimension = N,W,H,C
        pred_logits = pred_logits.permute(0, 2, 3, 1)
        pred_logits = pred_logits.numpy()

        #swap #plane
        pred_logits = plane_swap(pred_logits, plane=plane, inverse=True)
        #remove padding
        temp_logits = np.zeros((orig_shape[0], orig_shape[1], orig_shape[2], pred_logits.shape[3]))
        for i in range(pred_logits.shape[3]):
            temp_logits[:, :, :, i] = map_size(pred_logits[:, :, :, i], base_shape=(orig_shape[0], orig_shape[1], orig_shape[2]),verbose=0)

        return temp_logits

    def run_localization(self,fat_arr):

        localization_params=self.params_network.copy()
        localization_params['num_classes'] = 4
        model = CDFNet(localization_params)
        model.to(self.device)

        planes = ['coronal', 'sagittal']

        high_idx =0
        low_idx = 0
        orig_shape = fat_arr.shape

        for plane in planes:
            self.logger.info("--->Testing {} localization model".format(plane))
            # load model
            model_state = torch.load(self.flags['localization'][plane], map_location=self.device)
            model.load_state_dict(model_state["model_state_dict"])
            self.logger.info('Model weights loaded from {}'.format(self.flags['localization'][plane]))
            # organize data
            mod_arr = plane_swap(fat_arr, plane=plane)
            mod_arr = map_size(mod_arr, base_shape=(
            mod_arr.shape[0], self.params_network['width'], self.params_network['width']), verbose=0)
            mod_arr = mod_arr[..., np.newaxis]

            start = time.time()

            # evaluate
            logits = self.predict(mod_arr, batch_size=self.args.batch_size, model=model, plane=plane,
                                        orig_shape=orig_shape)

            pred = np.argmax(logits, axis=-1)

            temp = find_unique_index_slice(pred)

            high_idx += temp[0]
            low_idx += temp[1]

            end = time.time() - start
            self.logger.info("Tested Done in {:0.4f} seconds".format(end))

        high_idx = int(high_idx // 2)
        low_idx = int(low_idx // 2)

        return high_idx, low_idx

    # def run_view_agg(self,logit_pred):
    #     model = view_agg(self.view_agg_network.copy())
    #     model.to(self.device)
    #
    #     model_state = torch.load(self.flags['segmentation']['view_agg'], map_location=self.device)
    #     model.load_state_dict(model_state["model_state_dict"])
    #     self.logger.info('Model weights loaded from {}'.format(self.flags['segmentation']['view_agg']))
    #
    #     tx = ToTensorTest_logits()
    #     for key in logit_pred.keys():
    #             logit_pred[key] = plane_swap(logit_pred[key],'axial')
    #
    #     model.eval()
    #     with torch.no_grad():
    #         axial_batch = tx(logit_pred['axial'])
    #         axial_batch = axial_batch.unsqueeze(0)
    #         coronal_batch = tx(logit_pred['coronal'])
    #         coronal_batch = coronal_batch.unsqueeze(0)
    #         sagittal_batch = tx(logit_pred['sagittal'])
    #         sagittal_batch = sagittal_batch.unsqueeze(0)
    #
    #         if self.use_cuda:
    #             axial_batch, coronal_batch, sagittal_batch = axial_batch.cuda(), coronal_batch.cuda(), sagittal_batch.cuda()
    #
    #         pred_logits = model(axial_batch, coronal_batch, sagittal_batch)
    #
    #         pred_logits = pred_logits.squeeze(dim=0).cpu()
    #
    #         pred_logits = pred_logits.permute(1, 2, 3, 0)
    #         _, y_pred = torch.max(pred_logits, dim=-1)
    #
    #         # Change to Numpy
    #         y_pred = y_pred.numpy()
    #         y_pred = plane_swap(y_pred, plane='axial', inverse=True)
    #
    #     return y_pred

    def run_view_agg(self,logit_pred):
        pred = []
        for key in logit_pred.keys():
            logits = logit_pred[key][..., np.newaxis]
            pred.append(logits)

        pred_arr=np.concatenate(pred,axis=-1)
        pred_arr=np.sum(pred_arr,axis=-1)
        pred_arr=np.argmax(pred_arr,axis=-1)

        return pred_arr


    def run_segmentation(self,fat_arr):

        model= CDFNet(self.params_network.copy())
        model.to(self.device)

        if self.args.axial:
            self.logger.info('-' * 30)
            self.logger.info('Segmentation done only on the axial plane')
            self.logger.info('-' * 30)
            planes = ['axial']
        else:
            planes = ['axial','coronal','sagittal']

        logit_pred = {}
        orig_shape = fat_arr.shape

        for plane in planes:
            self.logger.info("--->Testing {} segmentation model".format(plane))
            #load model
            model_state = torch.load(self.flags['segmentation'][plane],map_location=self.device)
            model.load_state_dict(model_state["model_state_dict"])
            self.logger.info('Model weights loaded from {}'.format(self.flags['segmentation'][plane]))

            # organize data
            mod_arr = plane_swap(fat_arr, plane=plane)
            mod_arr = map_size(mod_arr, base_shape=(mod_arr.shape[0], self.params_network['width'], self.params_network['width']),verbose=0)
            mod_arr = mod_arr[..., np.newaxis]

            start = time.time()
            # evaluate
            logit_pred[plane] = self.predict(mod_arr, batch_size=self.args.batch_size, model=model,plane=plane,orig_shape=orig_shape)
            end = time.time() - start
            self.logger.info("Tested Done in {:0.4f} seconds".format(end))


        if self.args.axial:
            pred_arr =np.argmax(logit_pred['axial'],axis=-1)

        else:
            start = time.time()
            self.logger.info("--->Testing view aggregation model")
            pred_arr = self.run_view_agg(logit_pred)
            end = time.time() - start
            self.logger.info("Tested Done in {:0.4f} seconds".format(end))


        return pred_arr


    def eval(self, fat_arr):

        if self.args.run_localization:
            self.logger.info(30*'-')
            self.logger.info('Running localization models')
            high_idx, low_idx = self.run_localization(fat_arr)
            self.logger.info('ROI between slices %d, %d on the axial view' % (low_idx, high_idx))
        else:
            high_idx = fat_arr.shape[2]
            low_idx = 0

        self.logger.info(30 * '-')
        self.logger.info('Running segmentation models')
        pred_arr = self.run_segmentation(fat_arr)

        pred_arr[:,:,0:low_idx] = 0
        pred_arr[:,:,high_idx:] = 0

        if self.args.cleanup:
            self.logger.info('Cleaning segmentation')
            # Clean Segmentation Quick fixes
            mod_pred=np.zeros_like(pred_arr)
            sub_mask=get_largest_cc(pred_arr == 1)
            mod_pred[sub_mask] = 1
            other_mask=get_largest_cc(pred_arr == 4)
            mod_pred[other_mask] = 4
            mod_pred[pred_arr==2] = 2
            mod_pred[pred_arr==3] = 3

            # Remove small areas
            mask=remove_small_regions(mod_pred > 0,tolerance=25)
            pred_arr = pred_arr * mask

        return pred_arr, high_idx, low_idx









