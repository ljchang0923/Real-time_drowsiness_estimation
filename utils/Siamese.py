import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random

class SCCNet(nn.Module):
    def __init__(self, EEG_ch=30):
        super(SCCNet, self).__init__()

        # structure parameters
        self.num_ch = EEG_ch
        self.fs = 250
        self.F1 = 22
        self.F2 = 20
        self.t1 = self.fs // 10
        self.t2 = self.fs * 3

        # temporal and spatial filter
        self.Conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.F1, kernel_size=(self.num_ch, 1)),
            # nn.BatchNorm2d(22)
        )
        self.Conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.F2, kernel_size=(self.F1, self.t1), padding=(0, self.t1//2)),
            # nn.BatchNorm2d(20)
        )

        self.AveragePooling1 = nn.AvgPool2d((1, self.t2))
        # self.AveragePooling1 = nn.AvgPool2d((1,125), stride = (1,25))
        # stride is set as 25 (0.1 sec correspond to time domain)
        # kernel size 125 mean 0.5 sec
        self.regressor = predictor(self.F2)
        self.sigmoid = nn.Sigmoid()

    def square(self, x): 
        return torch.pow(x,2)

    def log_activation(self, x):
        return torch.log(x)

    def forward(self, x):
        x = self.Conv_block1(x)
        x = x.permute(0,2,1,3)
        # print(f'shape after depthwise separable conv: {x.size()}')
        x = self.Conv_block2(x)
        x = self.square(x)
        #print(f'shape after conv2: {x.size()}')
        x = x.permute(0,2,1,3)
        x = self.AveragePooling1(x)
        latent = self.log_activation(x)
        # print(f'shape before flatten: {x.size()}')
        x = torch.flatten(latent,1)
        # print(f'reshape: {x.size()}')
        x = self.regressor(x)
        output = self.sigmoid(x)
        return latent, output

class predictor(nn.Module):
    def __init__(self, fc2):
        super(predictor, self).__init__()

        self.regressor = nn.Sequential(
            # nn.Linear(fc1, fc2),
            # nn.ReLU(),
            nn.Linear(fc2, 1, bias = True)
            # nn.Sigmoid()
        )
    def forward(self, x):
        return self.regressor(x)

class Siamese_SCC(nn.Module):
    def __init__(self, EEG_ch=30, num_smooth=10):
        super(Siamese_SCC, self).__init__()

        self.sm_num = num_smooth
        self.feat_extractor = SCCNet(EEG_ch)
        self.dim_latent = self.feat_extractor.F2

        self.GAP = nn.AvgPool2d((1,self.sm_num))
        self.conv_fusion = nn.Conv2d(1, 1, (1, self.sm_num), bias=False)

        self.regressor = predictor(self.dim_latent) ## SCCNet: 20 EEGNet: 32 shallowConvNet: 40
        self.delta_regressor = nn.Sequential(
            nn.Linear(self.dim_latent*1, 1, bias = True)
        )
        ## Activation
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        ### Parallel sub-network 2
        latent = []
        for j in range(self.sm_num, x.size()[1]):
            t, _ = self.feat_extractor(x[:,j,:,:].unsqueeze(1))
            latent.append(t)
            
        latent = torch.cat(latent, 3)
        # print("latent size: ", latent.size())
        latent = self.GAP(latent)
        # latent = self.conv_fusion(latent)

        x_di = torch.flatten(latent, 1)
        x_di = self.regressor(x_di)
        output_di = self.sigmoid(x_di)
        
        ### Parallel sub-network 2
        with torch.no_grad():
            b_latent = []
            for i in range(0,self.sm_num):
                b, _ = self.feat_extractor(x[:,i,:,:].unsqueeze(1))
                b_latent.append(b)

            b_latent = torch.cat(b_latent,3)
            b_latent = self.GAP(b_latent)
            # b_latent = self.conv_fusion(b_latent)
            
        ### Regression head
        # x_delta = torch.cat((b_latent, latent), 2)
        x_delta = torch.sub(latent, b_latent)
        x_delta = torch.flatten(x_delta, 1)
        output_delta = self.delta_regressor(x_delta)
        output_delta = self.tanh(output_delta)
        # print('output size', output.size())

        return x_delta, output_di, output_delta