import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.encoder = nn.ModuleList([ 
            nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                ),
            
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                ),
            
            
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                ),
            
            nn.Sequential(
                nn.Conv2d(128,256, kernel_size=3, stride=2, padding=1, dilation=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                ),
            
            
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                ),
        ])
        
        self.decoder = nn.ModuleList([
        
            nn.Sequential(
                nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),),
        
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                ),
            
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                ),
            
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                ),
        
            
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                )
        ])
        
        # Final output layer to have a 30 output classes
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)
        # self.bnd1    = nn.BatchNorm2d(32)
        
        # self.conv2   = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        # self.bnd2    = nn.BatchNorm2d(64)
        
        # self.conv3   = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        # self.bnd3    = nn.BatchNorm2d(128)
        
        # self.conv4   = nn.Conv2d(128,256, kernel_size=3, stride=2, padding=1, dilation=1)
        # self.bnd4    = nn.BatchNorm2d(256)
        
        # self.conv5   = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1)
        # self.bnd5    = nn.BatchNorm2d(512)
        
        # self.relu    = nn.ReLU(inplace=True)
        
        # self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        # self.bn1     = nn.BatchNorm2d(512)
        
        # self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        # self.bn2     = nn.BatchNorm2d(256)
        
        # self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        # self.bn3     = nn.BatchNorm2d(128)
        
        # self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        # self.bn4     = nn.BatchNorm2d(64)
        
        # self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        # self.bn5     = nn.BatchNorm2d(32)
        # self.classifier = nn.Conv2d(32, 3, kernel_size=1)
        

        
        

    def forward(self, x):
        # Encode with ReLu after each convolution
        # out_encoder = self.bnd5(self.relu(self.conv5(
        #     self.bnd4(self.relu(self.conv4(
        #     self.bnd3(self.relu(self.conv3(
        #     self.bnd2(self.relu(self.conv2(
        #     self.bnd1(self.relu(self.conv1(x)))))))))))))))
        # Encode without ReLu after each convolution
        # x1 =    self.relu(
        #     self.bnd5(self.conv5(
        #     self.bnd4(self.conv4(
        #     self.bnd3(self.conv3(
        #     self.bnd2(self.conv2(
        #     self.bnd1(self.conv1(x)))))))))))
        
        # Decode with ReLu after each convolution
        # out_decoder = self.bn5(self.relu(self.deconv5(
        #     self.bn4(self.relu(self.deconv4(
        #     self.bn3(self.relu(self.deconv3(
        #     self.bn2(self.relu(self.deconv2(
        #     self.bn1(self.relu(self.deconv1(out_encoder)))))))))))))))
        
        out_encoder = x
        print('xshape: ', x.size())
        for layer in self.encoder:
            # Can use this to append to layers on reverse operations
            out_encoder = layer(out_encoder)
#             out_encoder = nn.MaxPool2d(2,2)(out_encoder)
            print('in encoder: ', out_encoder.size())
            
        print('out_encoder size: ', out_encoder.size())
            
        
        out_decoder = out_encoder
        
        for layer in self.decoder:
            out_decoder = layer(out_decoder)
            print('in decoder: ', out_decoder.size())

        print('out_decoder size: ', out_decoder.size())
        
        score = self.classifier(out_decoder)  
        print('score size: ', score.size())

        return score  # size=(N, n_class, x.H/1, x.W/1)