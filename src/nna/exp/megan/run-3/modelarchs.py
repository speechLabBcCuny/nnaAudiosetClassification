import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, maxMelLen, sampling_rate,device):
        # sr = 44100 etc
        self.maxMelLen = maxMelLen
        self.sampling_rate = sampling_rate
        torchaudio.set_audio_backend("sox_io")
        
        self.device=device
        #https://github.com/PCerles/audio/blob/3803d0b27a4e13efa760227ef6c71d0f3753aa98/test/test_transforms.py#L262
        #librosa defaults
        n_fft = 2048
        hop_length = 512
        power = 2.0
        n_mels = 128
        n_mfcc = 40
        # htk is false in librosa, no setting in torchaudio -?
        # norm is 1 in librosa, no setting in torchaudio -?
        self.melspect_transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.sampling_rate, window_fn=torch.hann_window,
                                                                  hop_length=hop_length, n_mels=n_mels, n_fft=n_fft).to(device)

    
        self.db_transform = torchaudio.transforms.AmplitudeToDB("power", 80.).to(device)
        
    def __call__(self, sample):
        x, y = sample
        x=x.to(self.device)
        mel = self.melspect_transform(x.reshape(-1))
        an_x = self.db_transform(mel)
        #librosa version
#         mel = librosa.feature.melspectrogram(y=x.reshape(-1),
#                                              sr=self.sampling_rate)
#         an_x = librosa.power_to_db(mel, ref=np.max)
#         an_x = an_x.astype("float32")
#         y = y.astype('float32')
#         print(an_x.shape)
        an_x = an_x[:, :self.maxMelLen]
        # 2-d conv
#         x = an_x.reshape(1, *an_x.shape[:])
        # 1-d conv
        x = an_x.reshape(1, an_x.shape[0]*an_x.shape[1])

        
        return x,y



# #test
# maxMelLen_test = 850
# SAMPLING_RATE_test = 48000
# sample_len_seconds = 10
# # to_tensor works on single sample
# sample_count = 1
# xx_test = torch.ones((sample_count,SAMPLING_RATE_test*sample_len_seconds))
# y_values = torch.ones(sample_count)
#
# toTensor = ToTensor(maxMelLen_test,SAMPLING_RATE_test)
# x_out,y_out=toTensor((xx_test,y_values))
# x_out.shape,y_out.shape


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


# mel.shape,an_x.shape,X_train.shape
class singleconv1dModel(nn.Module):
    '''A simple model for testing by overfitting.
    '''

    def __init__(self,
                 out_channels,
                 h_w,
                 kernel_size,
                 fc_1_size,
                 FLAT=False,
                 output_shape=(10,)):
        # h_w: height will be always one since we use 1d convolution
        super(singleconv1dModel, self).__init__()
        self.out_channels = out_channels
        #### CONV
        self.conv1 = nn.Conv1d(
            in_channels=1,  # depth of image == depth of filters
            out_channels=self.out_channels,  # number of filters 
            kernel_size=kernel_size,  # size of the filters/kernels
            padding=1)

        self.conv1_shape = conv_output_shape(h_w,
                                             kernel_size=kernel_size,
                                             stride=1,
                                             pad=1,
                                             dilation=1)
        # conv is 1d
        self.conv1_shape = (1, self.conv1_shape[1])

        self.fc1 = nn.Linear(self.out_channels * self.conv1_shape[0] *
                             self.conv1_shape[1], fc_1_size)  # 100

        self.fc2 = nn.Linear(fc_1_size, output_shape[0])

    def forward(self, x):
        #         x = x.reshape(1,)
        #         print(x.shape) #  50,1,108800 (850*128)
        x = F.relu(self.conv1(x))
        #         x = self.pool(x)
        # x = self.drop(x)
        #         print(x.shape)# 58, 2, 108801
        #         print(self.conv1_shape)
        #         print(x.shape)
        x = x.view(
            -1, self.out_channels * self.conv1_shape[0] * self.conv1_shape[1])
        # batch_norm is missing
        x = F.relu((self.fc1(x)))
        x = (self.fc2(x))

        #         x = self.drop(x)

        #         x = self.fc4(x)
        #         x = torch.sigmoid(x)
        #                 x = F.log_softmax(x,dim=1)
        return x


# test
# input_shape=(1,(938*128))
# output_shape=(10,)
# testModel_ins=adam(out_channels=2,h_w=input_shape,kernel_size=2,output_shape=output_shape)
# # a.conv1.weight
# a_out=testModel_ins(torch.ones((3,1,input_shape[1])))

# a_out_correct=torch.zeros(a_out.shape)
# a_out_correct[0][:]=1
# a_out_correct
# a_out.detach().numpy()

# torch.exp(a_out),a_out