import torch
from torch import nn
from utils.helper_funcs import default
from functools import partial
from blocks.common_blocks import Upsample, Downsample, Residual, PreNorm
from modules.embeddings import SinusoidalPositionEmbeddings
from blocks.resnet_blocks import ResnetBlock
from blocks.attentions import Attention, LinearAttention


__all__ = ["DermoSegDiff",]



class RB(nn.Module):
    """Some Information about RB"""
    def __init__(self, dim_in, dim_out, time_dim, resnet_block_groups=8):
        super().__init__()
        self.rb = ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim, groups=resnet_block_groups)

    def forward(self, x, t):
        return self.rb(x, t)


class LAtt(nn.Module):
    """Some Information about LAtt"""
    def __init__(self, dim):
        super().__init__()
        self.att = Residual(PreNorm(dim, LinearAttention(dim)))
    def forward(self, x):
        return self.att(x)


class LSAtt(nn.Module):
    '''Linear-Self Attention
    '''
    def __init__(self, dim):
        super().__init__()
        self.s_attn = Residual(PreNorm(dim, Attention(dim)))
        self.l_attn = LAtt(dim)
        
    def forward(self, x):
        x_s = self.s_attn(x)
        x_l = self.l_attn(x)
        return x_s + x_l


class BM(nn.Module):
    '''Bottleneck Module
    '''
    def __init__(self, dim_in, dim_out, time_dim, resnet_block_groups=4):
        super().__init__()
                
        self.rb_before = RB(dim_in, dim_out, time_dim, resnet_block_groups)
        self.ls_attn = LSAtt(dim_out)
        self.rb_after = RB(2*dim_out, dim_out, time_dim, resnet_block_groups)

    def forward(self, x, t):
        x = self.rb_before(x, t)
        x_a = self.ls_attn(x)
        x = torch.cat((x, x_a), dim=1)
        x = self.rb_after(x, t)
        return x


class EM(nn.Module):
    """Some Information about EM"""
    def __init__(self, dim_x, dim_g, time_x, time_g, resnet_block_groups=8):
        super().__init__()
        
        self.rb_x1 = RB(dim_x, dim_x, time_x, resnet_block_groups)
        self.rb_g1 = RB(dim_g, dim_g, time_g, resnet_block_groups)
        self.rb_x2 = RB(dim_x, dim_x, time_x, resnet_block_groups)
        self.rb_g2 = RB(dim_g+dim_x, dim_g, time_g, resnet_block_groups)
        self.g_feedback = nn.Conv2d(dim_g, dim_x, 1) # g_att_for_x
        self.att_x = LAtt(dim_x)
        self.att_g = LAtt(dim_g+dim_x)

    def forward(self, x, g, t_x, t_g):
        
        x = self.rb_x1(x, t_x)
        g = self.rb_g1(g, t_g)
        
        g = torch.cat((x, g), dim=1)
        h = g.clone()
        
        g = self.rb_g2(g, t_g)
        x = self.rb_x2(self.g_feedback(g) * x, t_x)
        
        x = self.att_x(x)
        g = torch.cat((x, g), dim=1)
        b = g.clone()
        
        g = self.att_g(g)
        
        outputs = (x, g)
        skips = (h, b)
        return outputs, skips


class DM(nn.Module):
    """Some Information about DM"""
    def __init__(self, dim_in, dim_out, time_dim, resnet_block_groups=8):
        super().__init__()
        
        self.rb_d1 = RB(dim_in, dim_out, time_dim, resnet_block_groups)
        self.rb_d2 = RB(dim_in, dim_out, time_dim, resnet_block_groups)
        self.att_x = LAtt(dim_out)
            
    def forward(self, x, b, h, t):
        x = torch.cat((x, b), dim=1)
        x = self.rb_d1(x, t)

        x = torch.cat((x, h), dim=1)
        x = self.rb_d2(x, t)
        x = self.att_x(x)
        return x



class DermoSegDiff(nn.Module):
    def __init__(
        self,
        dim_x, dim_g,
        channels_x=1, channels_g=3,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        dim_x_mults=None,
        dim_g_mults=None,
        resnet_block_groups=4,
        **kwargs
    ):
        '''
        `x` and `g` refer to target and guidance correspondingly.
        '''
        super().__init__()

        # determine dimensions
        init_dim_x = default(init_dim, dim_x)
        init_dim_g = default(init_dim, dim_g)
        
        dims_x = [init_dim_x, *map(lambda m: dim_x * m, default(dim_x_mults, dim_mults))]
        dims_g = [init_dim_g, *map(lambda m: dim_g * m, default(dim_g_mults, dim_mults))]
        
        in_out_x = list(zip(dims_x[:-1], dims_x[1:]))
        in_out_g = list(zip(dims_g[:-1], dims_g[1:]))
        
        num_resolutions = len(in_out_x)

        self.encoder_blocks = nn.ModuleList([])
        self.decoder_blocks = nn.ModuleList([])
        
        # time embeddings
        time_dim_x = dim_x * 4
        self.time_mlp_x = nn.Sequential(
            SinusoidalPositionEmbeddings(dim_x),  # (batch,dim)
            nn.Linear(dim_x, time_dim_x),
            nn.GELU(),
            nn.Linear(time_dim_x, time_dim_x),
        )
        time_dim_g = dim_g * 4
        self.time_mlp_g = nn.Sequential(
            SinusoidalPositionEmbeddings(dim_g),  # (batch,dim)
            nn.Linear(dim_g, time_dim_g),
            nn.GELU(),
            nn.Linear(time_dim_g, time_dim_g),
        )
        
        # initial steps
        self.init_conv_x = nn.Conv2d(channels_x, init_dim_x, 1, padding=0)
        self.init_conv_g = nn.Conv2d(channels_g, init_dim_g, 1, padding=0)
        
        # building encoder
        for ind, ((dim_in_x, dim_out_x), (dim_in_g, dim_out_g)) in enumerate(zip(in_out_x, in_out_g)):
            is_last = ind >= (num_resolutions - 1)
            
            encoder = EM(dim_x=dim_in_x, dim_g=dim_in_g, time_x=time_dim_x, time_g=time_dim_g, resnet_block_groups=resnet_block_groups)
            g_down = nn.Conv2d(dim_in_g+dim_in_x, dim_out_g, 3, padding=1) if is_last else Downsample(dim_in_g+dim_in_x, dim_out_g)
            x_down = nn.Conv2d(dim_in_x, dim_out_x, 3, padding=1) if is_last else Downsample(dim_in_x, dim_out_x)
            self.encoder_blocks.append(nn.ModuleList([encoder, g_down, x_down]))
    
        # building bottleneck
        self.bottleneck = BM(dim_in=dims_x[-1]+dims_g[-1], dim_out=dims_x[-1], time_dim=time_dim_x)

        # building decoder
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out_x)):
            is_last = ind == (len(in_out_x) - 1)
            skip_in = list(reversed(in_out_g))[ind][0] + dim_in
            
            decoder = DM(dim_in=skip_in+dim_out, dim_out=dim_out, time_dim=time_dim_x, resnet_block_groups=resnet_block_groups)
            d_up = nn.Conv2d(dim_out, dim_in, 3, padding=1) if is_last else Upsample(dim_out, dim_in)        
            self.decoder_blocks.append(nn.ModuleList([decoder, d_up]))
        
        # final steps
        self.final_rb = RB(2*dim_x+dim_g, dim_x, time_dim_x, resnet_block_groups=1)
        self.final_conv = nn.Conv2d(dim_x, default(out_dim, channels_x), 1)


    def forward(self, x, g, time):
        x = self.init_conv_x(x)
        r_x = x.clone()
        
        g = self.init_conv_g(g)
        r_g = g.clone()

        t_x = self.time_mlp_x(time)
        t_g = self.time_mlp_g(time)

        skips = []
        for (encoder, g_down, x_down) in self.encoder_blocks:
            (x, g), (h, b) = encoder(x, g, t_x, t_g)
            skips.append((h, b))
            g = g_down(g)
            x = x_down(x)
        
        x = self.bottleneck(torch.cat((x, g), dim=1), t_x)
        
        for (decoder, d_up) in self.decoder_blocks:
            (h, b) = skips.pop()
            x = decoder(x, b, h, t_x)
            x = d_up(x)
        
        x = self.final_rb(torch.cat((x, r_x, r_g), dim=1), t_x)
        return self.final_conv(x)

