import torch.nn as nn
from loguru import logger

class Encoder_Block(nn.Module):
    def __init__ (self,num_layers=2,in_channel=3,initial_filter=64):
        super(Encoder_Block,self).__init__()
        self.relu = nn.ReLU()
        self.layers = []
        out_channel = initial_filter
        for i in range(num_layers):
            self.layers.append(nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=1))
            self.layers.append(nn.ReLU())
            in_channel=out_channel
        self.block = nn.Sequential(*self.layers)

    def forward(self,x):
        x = self.block(x)
        return x