import torch.nn as nn
import torch.nn.functional as F
import torch

class Custom(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Sequential(
                nn.Conv2d(3,6, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm2d(6),
                self.relu,
                
                nn.Conv2d(6,6, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm2d(6),
                self.relu,
                
        )
        
        self.layer2 = nn.Sequential(
                nn.Conv2d(9,9, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm2d(9),
                self.relu,
                
                nn.Conv2d(9,9, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm2d(9),
                self.relu,
                
        )
        
        self.layer3 = nn.Sequential(
                nn.Conv2d(18,18, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm2d(18),
                self.relu,
                
                nn.Conv2d(18,18, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm2d(18),
                self.relu,
                
        )
        
        self.layer4 = nn.Sequential(
                nn.Conv2d(36,36, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm2d(36),
                self.relu,
                
                nn.Conv2d(36,36, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm2d(36),
                self.relu,
                
        )
        
        
        
        
        
        # Final output layer to have a 30 output classes
        self.layer5 = nn.Sequential(
                nn.Conv2d(72,72, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm2d(72),
                self.relu,
            
                nn.Conv2d(72,72, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm2d(72),
                self.relu,
                
                
                
        ) 
        
        self.classifier = nn.Conv2d(144, self.n_class, kernel_size=1)
        

        
        

    def forward(self, x):
        rep = self.layer1(x)
        
        x = torch.cat([x, rep], axis=1)
        rep = self.layer2(x)
        
        x = torch.cat([x, rep], axis=1)
        rep = self.layer3(x)
        
        x = torch.cat([x, rep], axis=1)
        rep = self.layer4(x)
        
        x = torch.cat([x, rep], axis=1)
        rep = self.layer5(x)
        
        x = torch.cat([x, rep], axis=1) 
        pred = self.classifier(x)
        del x
        del rep
        return pred