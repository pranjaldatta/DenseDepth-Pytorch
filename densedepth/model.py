import torch  
import torch.nn as nn  
import torch.nn.functional as F 
import torchvision.models as models  

class Encoder(nn.Module):

    def __init__(self, encoder_pretrained=True):
        super(Encoder, self).__init__() 
        self.densenet = models.densenet161(pretrained=encoder_pretrained)
    
    def forward(self, x):
        
        feature_maps = [x]

        for key, value in self.densenet.features._modules.items():
            feature_maps.append(value(feature_maps[-1]))
        
        return feature_maps

class Upsample(nn.Module):

    def __init__(self, input_channels, output_channels):

        super(Upsample, self).__init__() 

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.convA = nn.Conv2d(input_channels, output_channels, 3, 1, 1)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        self.convB = nn.Conv2d(output_channels, output_channels, 3, 1, 1)

    def forward(self, x, concat_with):

        concat_h_dim = concat_with.shape[2]
        concat_w_dim = concat_with.shape[3]

        upsampled_x = F.interpolate(x, size=[concat_h_dim, concat_w_dim], mode="bilinear", align_corners=True)
        upsampled_x = torch.cat([upsampled_x, concat_with], dim=1)
        
        upsampled_x = self.convA(upsampled_x)
        upsampled_x = self.leakyrelu(upsampled_x)
        upsampled_x = self.convB(upsampled_x)
        upsampled_x = self.leakyrelu(upsampled_x)

        return upsampled_x


class Decoder(nn.Module):

    def __init__(self, num_features=2208, decoder_width=0.5, scales=[1, 2, 4, 8]):

        super(Decoder, self).__init__()

        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, 1, 1, 1)

        self.upsample1 = Upsample(features//scales[0] + 384, features//(scales[0] * 2))
        self.upsample2 = Upsample(features//scales[1] + 192, features//(scales[1] * 2))
        self.upsample3 = Upsample(features//scales[2] + 96, features//(scales[2] * 2))
        self.upsample4 = Upsample(features//scales[3] + 96, features//(scales[3] * 2))

        self.conv3 = nn.Conv2d(features//(scales[3] * 2), 1, 3, 1, 1)

    def forward(self, features):

        x_block0= features[3]
        x_block1 = features[4]
        x_block2 = features[6]
        x_block3 = features[8]
        x_block4 = features[11]

        x0 = self.conv2(x_block4)
        x1 = self.upsample1(x0, x_block3)
        x2 = self.upsample2(x1, x_block2)
        x3 = self.upsample3(x2, x_block1)
        x4 = self.upsample4(x3, x_block0)

        return self.conv3(x4)

class DenseDepth(nn.Module):

    def __init__(self, encoder_pretrained=True):

        super(DenseDepth, self).__init__()

        self.encoder = Encoder(encoder_pretrained=encoder_pretrained)
        self.decoder = Decoder()
    
    def forward(self, x):
        
        x = self.encoder(x)
        x = self.decoder(x)

        return x