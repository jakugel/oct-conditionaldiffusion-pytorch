import torch.nn as nn
import torch
import numpy as np


class ResidualConvBlock(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_type='transpose'):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        if upsample_type == 'transpose':
            upsample_layers = [nn.ConvTranspose2d(in_channels, out_channels, 2, 2)]
        elif upsample_type == 'upconv':
            upsample_layers = [nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(in_channels, out_channels, 3, 1, 'same')]

        layers = upsample_layers + [
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, num_layers=2, n_classes=10, image_size=32, upsample_type='transpose'):
        super(ContextUnet, self).__init__()

        self.image_levels = np.log2(image_size)

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes
        self.num_layers = num_layers  # Store number of layers for dynamic creation

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        # Dynamically create downsampling layers
        self.down_blocks = nn.ModuleList()
        for i in range(num_layers):
            in_channels = 2 ** max((i - 1), 0) * self.n_feat
            out_channels = n_feat * (2**i)  # Double the channels at each layer
            self.down_blocks.append(UnetDown(in_channels, out_channels))

        self.to_vec = nn.Sequential(nn.AvgPool2d(2 ** int(self.image_levels - self.num_layers)), nn.GELU())

        # Dynamically create embedding layers
        self.timeembeds = nn.ModuleList([EmbedFC(1, 2 ** (self.num_layers - 1 - i) * self.n_feat) for i in range(num_layers)])
        self.contextembeds = nn.ModuleList([EmbedFC(n_classes, 2 ** (self.num_layers - 1 - i) * self.n_feat) for i in range(num_layers)])

        # Dynamically create upsampling layers
        self.up_blocks = nn.ModuleList()
        for i in range(num_layers):
            in_channels = 2 ** (self.num_layers - i) * self.n_feat
            out_channels = 2 ** max((self.num_layers - i - 2), 0) * self.n_feat
            self.up_blocks.append(UnetUp(in_channels, out_channels, upsample_type=upsample_type))

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 ** (self.num_layers - 1) * self.n_feat, 2 ** (self.num_layers - 1) * self.n_feat, 2 ** int(self.image_levels - self.num_layers), 2 ** int(self.image_levels - self.num_layers)),
            nn.GroupNorm(2 ** int(self.image_levels - self.num_layers), 2 ** (self.num_layers - 1) * self.n_feat),
            nn.ReLU(),
        )

        self.out = nn.Sequential(
            nn.Conv2d(n_feat * 2, n_feat, 3, 1, 1),
            nn.GroupNorm(2 ** int(self.image_levels - self.num_layers), n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        x = self.init_conv(x)
        x_down = x
        # Downsampling loop for dynamic layers
        down_layers = []
        for down_block in self.down_blocks:
            x_down = down_block(x_down)
            down_layers.append(x_down)

        hiddenvec = self.to_vec(x_down)

        # Convert context to one-hot embedding
        c = nn.functional.one_hot(c, num_classes=self.n_classes).float()

        # Mask out context if context_mask == 1
        context_mask = context_mask[:, None].repeat(1, self.n_classes)
        context_mask = (-1 * (1 - context_mask))  # Flip 0 <-> 1
        c = c * context_mask

        # Embedding for each layer
        embeddings = []
        for i in range(self.num_layers):
            cemb = self.contextembeds[i](c).view(-1, 2 ** (self.num_layers - i - 1) * self.n_feat, 1, 1)
            temb = self.timeembeds[i](t).view(-1, 2 ** (self.num_layers - i - 1) * self.n_feat, 1, 1)
            embeddings.append((cemb, temb))

        up1 = self.up0(hiddenvec)
        for i in range(self.num_layers):
            cemb, temb = embeddings[i]
            up1 = self.up_blocks[i](cemb * up1 + temb, down_layers[self.num_layers - i - 1])

        out = self.out(torch.cat((up1, x), 1))
        return out