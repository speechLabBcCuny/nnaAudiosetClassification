import torch
import torch.nn as nn
import torch.nn.functional as F


class testModel(nn.Module):
    '''A simple model for testing by overfitting.
    '''
    def __init__(self, out_channels, h_w, kernel_size, FLAT=False):
        super(testModel, self).__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(in_channels=1,
                               out_channels=self.out_channels,
                               kernel_size=kernel_size,
                               padding=(1, 1))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # x = self.drop(x)

        x = x.view(-1, self.out_channels * self.conv1_h * self.conv1_w)

        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.drop(x)

        x = self.fc4(x)
        x = torch.sigmoid(x)
        #         x = F.log_softmax(x,dim=1)
        return x



class NetCNN1(nn.Module):
    '''model with single CNN layer
    '''
    def __init__(self, out_channels, h_w, kernel_size, FLAT=False):
        super(NetCNN1, self).__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=self.out_channels,
                               kernel_size=kernel_size,
                               padding=(1, 1))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_h, self.conv1_w = conv_output_shape(h_w,
                                                       kernel_size=kernel_size,
                                                       pad=1)
        # maxpool affect
        self.conv1_h, self.conv1_w = self.conv1_h // 2, self.conv1_w // 2
        #         print(self.conv1_h,self.conv1_w)
        # #         self.conv2 = nn.Conv2d(6, 16, 5)
        if FLAT:
            self.fc1 = nn.Linear(1280, 10)  # 100
        else:
            self.fc1 = nn.Linear(self.out_channels * self.conv1_h *
                                 self.conv1_w, 10)  # 100


#         self.fc1 = nn.Linear(128, 32)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        self.fc1_bn = nn.BatchNorm1d(10)  # 100

        self.fc2 = nn.Linear(10, 10)  # 32
        torch.nn.init.xavier_normal_(self.fc2.weight)
        self.fc2_bn = nn.BatchNorm1d(10)

        self.fc3 = nn.Linear(10, 10)
        torch.nn.init.xavier_normal_(self.fc3.weight)
        self.fc3_bn = nn.BatchNorm1d(5)

        self.fc4 = nn.Linear(10, 10)  # 100
        torch.nn.init.xavier_normal_(self.fc4.weight)

        self.drop = nn.Dropout(p=0.2)  # 0.2

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        #         x = self.pool(F.relu(self.conv2(x)))
        x = self.drop(x)

        x = x.view(-1, self.out_channels * self.conv1_h * self.conv1_w)

        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.drop(x)

        x = self.fc4(x)
        x = torch.sigmoid(x)
        #         x = F.log_softmax(x,dim=1)
        return x


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
  Utility function for computing output of convolutions
  takes a tuple of (h,w) and returns a tuple of (h,w)
  """
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor(((h_w[0] + (2 * pad) - (dilation *
                                      (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = floor(((h_w[1] + (2 * pad) - (dilation *
                                      (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


class NetCNN2(nn.Module):

    def __init__(self,
                 out_channels,
                 out_channels2,
                 h_w,
                 kernel_size,
                 kernel_size2,
                 FLAT=False):
        super(NetCNN2, self).__init__()
        self.out_channels = out_channels
        self.out_channels2 = out_channels2
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=self.out_channels,
                               kernel_size=kernel_size,
                               padding=(1, 1))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_h, self.conv1_w = conv_output_shape(h_w,
                                                       kernel_size=kernel_size,
                                                       pad=1)
        self.conv1_h, self.conv1_w = self.conv1_h // 2, self.conv1_w // 2

        self.conv2 = nn.Conv2d(in_channels=self.out_channels,
                               out_channels=self.out_channels2,
                               kernel_size=kernel_size2,
                               padding=(1, 1))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2_h, self.conv2_w = conv_output_shape(
            [self.conv1_h, self.conv1_w], kernel_size=kernel_size2, pad=1)
        self.conv2_h, self.conv2_w = self.conv2_h // 2, self.conv2_w // 2

        #         print(self.conv1_h,self.conv1_w)
        # #         self.conv2 = nn.Conv2d(6, 16, 5)
        if FLAT:
            self.fc1 = nn.Linear(1280, 10)  # 100
        else:
            self.fc1 = nn.Linear(self.out_channels2 * self.conv2_h *
                                 self.conv2_w, 10)  # 100


#         self.fc1 = nn.Linear(128, 32)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        self.fc1_bn = nn.BatchNorm1d(10)  # 100

        self.fc2 = nn.Linear(10, 10)  # 32
        torch.nn.init.xavier_normal_(self.fc2.weight)
        self.fc2_bn = nn.BatchNorm1d(10)

        self.fc3 = nn.Linear(10, 10)
        torch.nn.init.xavier_normal_(self.fc3.weight)
        self.fc3_bn = nn.BatchNorm1d(5)

        self.fc4 = nn.Linear(10, 10)  # 100
        torch.nn.init.xavier_normal_(self.fc4.weight)

        self.drop = nn.Dropout(p=0.2)  # 0.2

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop(x)

        x = x.view(-1, self.out_channels2 * self.conv2_h * self.conv2_w)

        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.drop(x)

        x = self.fc4(x)
        x = torch.sigmoid(x)
        #         x = F.log_softmax(x,dim=1)
        return x
