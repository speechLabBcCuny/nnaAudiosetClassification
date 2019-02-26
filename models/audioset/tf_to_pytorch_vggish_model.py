import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict

class KitModel(nn.Module):

    
    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        global __weights_dict
        __weights_dict = load_weights(weight_file)

        self.vggish_conv1_Conv2D = self.__conv(2, name='vggish/conv1/Conv2D', in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.vggish_conv2_Conv2D = self.__conv(2, name='vggish/conv2/Conv2D', in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.vggish_conv3_conv3_1_Conv2D = self.__conv(2, name='vggish/conv3/conv3_1/Conv2D', in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.vggish_conv3_conv3_2_Conv2D = self.__conv(2, name='vggish/conv3/conv3_2/Conv2D', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.vggish_conv4_conv4_1_Conv2D = self.__conv(2, name='vggish/conv4/conv4_1/Conv2D', in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.vggish_conv4_conv4_2_Conv2D = self.__conv(2, name='vggish/conv4/conv4_2/Conv2D', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.vggish_fc1_fc1_1_MatMul = self.__dense(name = 'vggish/fc1/fc1_1/MatMul', in_features = 12288, out_features = 4096, bias = True)
        self.vggish_fc1_fc1_2_MatMul = self.__dense(name = 'vggish/fc1/fc1_2/MatMul', in_features = 4096, out_features = 4096, bias = True)
        self.vggish_fc2_MatMul = self.__dense(name = 'vggish/fc2/MatMul', in_features = 4096, out_features = 128, bias = True)

    def forward(self, x):
        self.vggish_Flatten_flatten_Reshape_shape_1 = torch.autograd.Variable(torch.Tensor([-1]), requires_grad=False)
        vggish_Reshape  = torch.reshape(input = x, shape = (1,96,64,1)).permute(0, 3, 1, 2)

        vggish_conv1_Conv2D_pad = F.pad(vggish_Reshape, (1, 1, 1, 1))
        vggish_conv1_Conv2D = self.vggish_conv1_Conv2D(vggish_conv1_Conv2D_pad)
        vggish_conv1_Relu = F.relu(vggish_conv1_Conv2D)
        vggish_pool1_MaxPool = F.max_pool2d(vggish_conv1_Relu, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        vggish_conv2_Conv2D_pad = F.pad(vggish_pool1_MaxPool, (1, 1, 1, 1))
        vggish_conv2_Conv2D = self.vggish_conv2_Conv2D(vggish_conv2_Conv2D_pad)
        vggish_conv2_Relu = F.relu(vggish_conv2_Conv2D)
        vggish_pool2_MaxPool = F.max_pool2d(vggish_conv2_Relu, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        vggish_conv3_conv3_1_Conv2D_pad = F.pad(vggish_pool2_MaxPool, (1, 1, 1, 1))
        vggish_conv3_conv3_1_Conv2D = self.vggish_conv3_conv3_1_Conv2D(vggish_conv3_conv3_1_Conv2D_pad)
        vggish_conv3_conv3_1_Relu = F.relu(vggish_conv3_conv3_1_Conv2D)
        vggish_conv3_conv3_2_Conv2D_pad = F.pad(vggish_conv3_conv3_1_Relu, (1, 1, 1, 1))
        vggish_conv3_conv3_2_Conv2D = self.vggish_conv3_conv3_2_Conv2D(vggish_conv3_conv3_2_Conv2D_pad)
        vggish_conv3_conv3_2_Relu = F.relu(vggish_conv3_conv3_2_Conv2D)
        vggish_pool3_MaxPool = F.max_pool2d(vggish_conv3_conv3_2_Relu, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        vggish_conv4_conv4_1_Conv2D_pad = F.pad(vggish_pool3_MaxPool, (1, 1, 1, 1))
        vggish_conv4_conv4_1_Conv2D = self.vggish_conv4_conv4_1_Conv2D(vggish_conv4_conv4_1_Conv2D_pad)
        vggish_conv4_conv4_1_Relu = F.relu(vggish_conv4_conv4_1_Conv2D)
        vggish_conv4_conv4_2_Conv2D_pad = F.pad(vggish_conv4_conv4_1_Relu, (1, 1, 1, 1))
        vggish_conv4_conv4_2_Conv2D = self.vggish_conv4_conv4_2_Conv2D(vggish_conv4_conv4_2_Conv2D_pad)
        vggish_conv4_conv4_2_Relu = F.relu(vggish_conv4_conv4_2_Conv2D)
        vggish_pool4_MaxPool = F.max_pool2d(vggish_conv4_conv4_2_Relu, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False).permute(0, 2, 3, 1)
        vggish_Flatten_flatten_Shape = torch.Tensor(list(vggish_pool4_MaxPool.size()))
        vggish_Flatten_flatten_Reshape = torch.reshape(input = vggish_pool4_MaxPool, shape = (1,12288))
        vggish_Flatten_flatten_strided_slice = vggish_Flatten_flatten_Shape[0:1][0]
        vggish_fc1_fc1_1_MatMul = self.vggish_fc1_fc1_1_MatMul(vggish_Flatten_flatten_Reshape)
        vggish_Flatten_flatten_Reshape_shape = [vggish_Flatten_flatten_strided_slice,self.vggish_Flatten_flatten_Reshape_shape_1]
        vggish_fc1_fc1_1_Relu = F.relu(vggish_fc1_fc1_1_MatMul)
        vggish_fc1_fc1_2_MatMul = self.vggish_fc1_fc1_2_MatMul(vggish_fc1_fc1_1_Relu)
        vggish_fc1_fc1_2_Relu = F.relu(vggish_fc1_fc1_2_MatMul)
        vggish_fc2_MatMul = self.vggish_fc2_MatMul(vggish_fc1_fc1_2_Relu)
        vggish_fc2_Relu = F.relu(vggish_fc2_MatMul)
        return vggish_fc2_Relu


    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer