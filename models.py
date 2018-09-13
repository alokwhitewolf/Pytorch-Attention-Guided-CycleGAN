import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    

    def __init__(self, in_features, norm=False):
        super(ResBlock, self).__init__()

        block = [  nn.ReflectionPad2d(1),
                nn.Conv2d(in_features, in_features, 3),
                # nn.InstanceNorm2d(in_features),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_features, in_features, 3),
                # nn.InstanceNorm2d(in_features)
                ]

        if norm:
            block.insert(2,  nn.InstanceNorm2d(in_features))
            block.insert(6,  nn.InstanceNorm2d(in_features))


        self.model = nn.Sequential(*block)

    def forward(self, x):
        return x + self.model(x)


class Gen(nn.Module):
    

    def __init__(self, input_nc=3, output_nc=3, n_resblocks=9, norm=False):
        super(Gen, self).__init__()
        
        model = [   nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 32, 7),
            nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 32
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_resblocks):
            model += [ResBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(32, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    
    def forward(self, x):
        return self.model(x)


class Dis(nn.Module):
    

    def __init__(self, input_nc):
        super(Dis, self).__init__()

        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class Attn(nn.Module):


    def __init__(self, input_nc=3):
        super(Attn, self).__init__()

        model =  [  nn.Conv2d(3, 32, 7, stride=1, padding=3),
                    nn.InstanceNorm2d(32),
                    nn.ReLU(inplace=True) ]

        model += [  nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        model += [ResBlock(64, norm=True)]

        model += [nn.UpsamplingNearest2d(scale_factor=2)]

        model += [  nn.Conv2d(64, 64, 3, stride=1, padding=1),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]
        
        # model += [nn.UpsamplingNearest2d(scale_factor=2)]

        model += [  nn.Conv2d(64, 32, 3, stride=1, padding=1),
                    nn.InstanceNorm2d(32),
                    nn.ReLU(inplace=True) ]

        model += [  nn.Conv2d(32, 1, 7, stride=1, padding=3),
                    nn.Sigmoid() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)








