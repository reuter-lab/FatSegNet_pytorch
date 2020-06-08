
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

# IMPORTS
import torch
import torch.nn as nn
from torchsummary import summary




class view_agg(nn.Module):
    """

    """
    def __init__(self, params):
        super(view_agg, self).__init__()

        # Padding to get output tensor of same dimensions
        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)
        padding_d= int((params['kernel_d'] - 1) / 2)

        self.conv0=nn.Conv3d(in_channels=3*params['num_channels'],out_channels=params['num_filters'],
                             kernel_size=(params['kernel_d'],params['kernel_h'],params['kernel_w']),
                             stride=params['stride_conv'],padding=(padding_d,padding_h,padding_w))

        self.conv1=nn.Conv3d(in_channels=params['num_filters'],out_channels=params['num_classes'],
                             kernel_size=(1,1,1),stride=params['stride_conv'],padding=(0,0,0))


        self.bn0=nn.BatchNorm3d(num_features=params['num_channels']*3)
        self.bn1=nn.BatchNorm3d(num_features=params['num_filters'])
        self.prelu = nn.PReLU()


        # Code for Network Initialization

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, axial,coronal,sagittal):
        """
        Computational graph
        :param tensor x: input image
        :return tensor: prediction logits
        """

        #concatenate around the channel axis
        x0=torch.cat((axial,coronal,sagittal),dim=1)

        x0 = self.bn0(x0)
        x0 = self.prelu(x0)
        x1 = self.conv0(x0)

        x1= self.bn1(x1)
        x1=self.prelu(x1)
        out=self.conv1(x1)

        return out



