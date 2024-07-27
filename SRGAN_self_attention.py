import torch
import torch.nn as nn
from ops import *
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1) 
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height) 
        energy = torch.bmm(proj_query, proj_key) 
        attention = self.softmax(energy) 
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out

class Generator(nn.Module):
    
    def __init__(self, img_feat=3, n_feats=64, kernel_size=3, num_block=16, act=nn.PReLU(), scale=4):
        super(Generator, self).__init__()
        
        self.conv01 = conv(in_channel=img_feat, out_channel=n_feats, kernel_size=9, BN=False, act=act)
        
        resblocks = [ResBlock(channels=n_feats, kernel_size=3, act=act) for _ in range(num_block)]
        self.body = nn.Sequential(*resblocks)
        
        self.conv02 = conv(in_channel=n_feats, out_channel=n_feats, kernel_size=3, BN=True, act=None)
        
        self.attention = SelfAttention(in_dim=n_feats)
        
        if scale == 4:
            upsample_blocks = [Upsampler(channel=n_feats, kernel_size=3, scale=2, act=act) for _ in range(2)]
        else:
            upsample_blocks = [Upsampler(channel=n_feats, kernel_size=3, scale=scale, act=act)]

        self.tail = nn.Sequential(*upsample_blocks)
        
        self.last_conv = conv(in_channel=n_feats, out_channel=img_feat, kernel_size=3, BN=False, act=nn.Tanh())
        
    def forward(self, x):
        
        x = self.conv01(x)
        _skip_connection = x
        
        x = self.body(x)
        x = self.conv02(x)
        x = self.attention(x)
        feat = x + _skip_connection
        
        x = self.tail(feat)
        x = self.last_conv(x)
        
        return x, feat
    
class Discriminator(nn.Module):
    
    def __init__(self, img_feat=3, n_feats=64, kernel_size=3, act=nn.LeakyReLU(inplace=True), num_of_block=3, patch_size=96):
        super(Discriminator, self).__init__()
        self.act = act
        
        self.conv01 = conv(in_channel=img_feat, out_channel=n_feats, kernel_size=3, BN=False, act=self.act)
        self.conv02 = conv(in_channel=n_feats, out_channel=n_feats, kernel_size=3, BN=False, act=self.act, stride=2)
        
        body = [discrim_block(in_feats=n_feats * (2 ** i), out_feats=n_feats * (2 ** (i + 1)), kernel_size=3, act=self.act) for i in range(num_of_block)]
        self.body = nn.Sequential(*body)
        
        # Adding SelfAttention layer after the body blocks
        self.attention = SelfAttention(n_feats * (2 ** num_of_block))
        
        self.linear_size = ((patch_size // (2 ** (num_of_block + 1))) ** 2) * (n_feats * (2 ** num_of_block))
        
        tail = []
        tail.append(nn.Linear(self.linear_size, 1024))
        tail.append(self.act)
        tail.append(nn.Linear(1024, 1))
        tail.append(nn.Sigmoid())
        
        self.tail = nn.Sequential(*tail)
        
    def forward(self, x):
        x = self.conv01(x)
        x = self.conv02(x)
        x = self.body(x)
        
        # Applying the SelfAttention layer
        x = self.attention(x)
        
        x = x.view(-1, self.linear_size)
        x = self.tail(x)
        
        return x
