from torchsummary import summary
from detect import parse_args
import torch.nn as nn 
import torch

args = parse_args()

def ConvLayer(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, use_batchnorm = True):
    layer = [
        nn.Conv2d(in_channels= in_channels, out_channels= out_channels, kernel_size= kernel_size, stride= stride,
                  padding= padding),
        nn.LeakyReLU(0.1)
    ]
     
    if use_batchnorm:
        layer.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layer)

def TinyYOLO(S=args.grid, BOX=args.box, CLS=args.cls):
    layers = nn.Sequential(
        ConvLayer(3, 16, kernel_size=3, stride=1, padding=1, use_batchnorm=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        ConvLayer(16, 32, kernel_size=3, stride=1, padding=1, use_batchnorm=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        ConvLayer(32, 64, kernel_size=3, stride=1, padding=1, use_batchnorm=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        ConvLayer(64, 128, kernel_size=3, stride=1, padding=1, use_batchnorm=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        ConvLayer(128, 256, kernel_size=3, stride=1, padding=1, use_batchnorm=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        ConvLayer(256, 512, kernel_size=3, stride=1, padding=1, use_batchnorm=True),

        ConvLayer(512, 1024, kernel_size=3, stride=1, padding=1, use_batchnorm=True),

        ConvLayer(1024, 1024, kernel_size=3, stride=1, padding=1, use_batchnorm=True),

        nn.Conv2d(1024, BOX * (5 + CLS), kernel_size=1, stride=1, padding=0)
    )
    return layers


class TINYMODEL(torch.nn.Module):
    def __init__(self, S=args.grid, BOX=args.box, CLS=args.cls):
        super(TINYMODEL, self).__init__()
        self.S = S
        self.BOX = BOX 
        self.CLS = CLS 
        self.layers = TinyYOLO(S, BOX, CLS)
        
    def forward(self, input):
        output_tensor = self.layers(input)
        output_tensor = output_tensor.permute(0, 2, 3, 1) 
        W_grid, H_grid = self.S, self.S
        output_tensor = output_tensor.view(-1, H_grid, W_grid, self.BOX, 4 + 1 + self.CLS)
        return output_tensor

