from torch import nn
import torch
from torch.nn import functional as F
import warnings
from timm.models.layers import to_2tuple


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(
            self, input_dim, hidden_dim, output_dim, num_layers, affine_func=nn.Linear
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            affine_func(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: torch.Tensor):
        for i, layer in enumerate(self.layers):
            # L R L R L
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class Fusion(nn.Module):
    def __init__(self, clip_dim, adapter_dim):
        super().__init__()
        self.clip_dim = clip_dim
        self.adapter_dim = adapter_dim
        self.proj = nn.Sequential(
            LayerNorm(clip_dim),
            nn.Conv2d(clip_dim, adapter_dim, kernel_size=1),
        )

    def forward(self, x, clip_x, spatial_shape):
        h, w = spatial_shape
        n, l, d = clip_x.shape

        if l == h*w:
            clip_x = clip_x.permute(0, 2, 1).view(n, d, h, w)  # NLD->NDL->NDhw
        else:
            clip_x = clip_x.permute(0, 2, 1).view(n, d, 14, 14)  # NLD->NDL->NDhw
            clip_x = F.interpolate(
                clip_x.contiguous(),
                size= (16, 16),
                mode="bilinear",
                align_corners=False,
            ) #ND 14 14 => N D 16 16
        clip_x = self.proj(clip_x).view(n, self.adapter_dim, h * w).permute(0, 2, 1)
        x = x + clip_x  # NLD

        return x



class MaskPostXrayProcess(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.process = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=in_c // 2, kernel_size=3, stride=1, padding=1),  # (N Q h,w)->(N 64 h,w))
            nn.BatchNorm2d(in_c // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_c // 2, out_channels=in_c // 4, kernel_size=3, stride=1, padding=1),  # (N 32 h,w)
            nn.BatchNorm2d(in_c // 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_c // 4, out_channels=1, kernel_size=1, stride=1, padding=0),  # (N 16 h,w)

            nn.ConvTranspose2d(in_channels=1, out_channels=1 , kernel_size=16, stride=16) # (N 16 h,w)->(N 1 256 256)
            #nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)  # (N 1 256 256)
        )


    def forward(self, x, if_boundaries):
        x = x.reshape(x.shape[0], x.shape[1], -1) #(N Q 256)
        x = x.permute(0, 2, 1)#(N L Q)
        if_boundaries = if_boundaries.unsqueeze(-1)  # (NL1) 不是boundry的patch块置为0

        x = x * if_boundaries #(N L Q) * (N L 1)
        x = x.permute(0, 2, 1) #(N Q L)
        x = x.reshape(x.shape[0], x.shape[1], 16, 16)

        post_x = self.process(x)  #(N 1 224 224)
        return post_x



class PostClipProcess(nn.Module):
    """
    NQD -> ND -> N2

    """

    def __init__(self, num_quires, embed_dim):
        super().__init__()

        self.first_process = nn.Sequential(
            nn.Conv1d(in_channels=num_quires, out_channels=num_quires // 2, kernel_size=3, stride=1, padding=1), # NQD->N1D
            nn.BatchNorm1d(num_quires // 2),
            nn.ReLU(),
            nn.Conv1d(in_channels=num_quires // 2, out_channels=num_quires // 4 , kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_quires // 4),
            nn.ReLU(),
            nn.Conv1d(in_channels=num_quires // 4, out_channels=1, kernel_size=3, stride=1, padding=1),

        )
        #self.norm = VT_LN(embed_dim)
        self.second_process = nn.Sequential(  # ND->N2
            nn.Linear(in_features=embed_dim, out_features=embed_dim // 2),
            nn.ReLU(),
            nn.Linear(in_features=embed_dim // 2, out_features=embed_dim // 4),
            nn.ReLU(),
            #nn.Linear(in_features=embed_dim // 4, out_features=embed_dim // 8),
            #nn.ReLU(),
            nn.Linear(in_features=embed_dim // 4, out_features=2)
        )


    def forward(self, x):
        x = self.first_process(x)  #NQD->N1D
        x = x.squeeze()
        x = self.second_process(x)
        return x



class VT_LN(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class PatchEmbed(nn.Module):

    def __init__(
            self,
            img_size=256,
            patch_size=16,
            in_chans=3,
            embed_dim=192,
            norm_layer=None,
            bias=False,
            **kwargs
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )

        self.norm = VT_LN(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # NDL
        x = x.permute(0, 2, 1)  # NDL->NLD
        # x = self.norm(x)
        return x
