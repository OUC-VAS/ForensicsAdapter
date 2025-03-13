from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from model.layer import MLP

class ClipIntraBlock(nn.Module):
    def __init__(self,num_features):
        super().__init__()
        self.num_features = num_features
        self.conv_first =nn.Conv1d(in_channels=self.num_features, out_channels=192, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv_second =nn.Conv1d(in_channels=192, out_channels=self.num_features, kernel_size=1)

    def forward(self, x, data_dict, clip_L, inference):

        intra_x = x.permute(1, 2, 0)  # LND -> NDL
        intra_x = intra_x[:, :, -clip_L:].float()  # N  clipD 256/196

        intra_x = self.conv_first(intra_x)  # N D 256
        intra_x = self.relu(intra_x)
        intra_x = intra_x.permute(0, 2, 1)  # NDL-> NLD
        if not inference:
            loss_clip = self.intra_contra(intra_x, data_dict['clip_patch_label'], data_dict['label'], (16, 16))
        else:
            loss_clip = 0
        intra_x = intra_x.permute(0, 2, 1)  # NLD-> NDL
        intra_x = self.conv_second(intra_x)  # NDL
        # intra_x = self.relu(intra_x)

        intra_x = intra_x.permute(2, 0, 1)  # NDL- >LND
        # x LND
        # x[-clip_L:,...] = intra_x * 0.1 + x[-clip_L:, ...] * self.intra_scale 在B14 效果不错
        # x[-clip_L:, ...] = intra_x * self.intra_scale + x[-clip_L:, ...] * 0.9
        #x[-clip_L:, ...] = intra_x * 0.15 + x[-clip_L:, ...] * 0.95
        return intra_x, loss_clip

    def intra_contra(self, x, patch_labels, image_labels, spatial_shape):
        if True:
            L = spatial_shape[0] * spatial_shape[1]  # 14 * 14 = 196  16 * 16=256

            embeddings = x[:, -L:, ...]  # (N 196 192) (NLD) (N 256 128)
            embeddings = nn.functional.normalize(embeddings, dim=-1)
            fake_index = (image_labels == 1).nonzero(as_tuple=True)[0]
            fake_embeddings = embeddings[fake_index]  # f_N L D

            ff_patch_labels = patch_labels[fake_index]  # f_N L
            fr_patch_labels = torch.logical_not(ff_patch_labels)

            # fake 图片的real和fake part
            f_fake_part = fake_embeddings * ff_patch_labels.unsqueeze(-1)  # f_N L D *  f_N L 1 -> f_N L D
            f_real_part = fake_embeddings * fr_patch_labels.unsqueeze(-1)  # f_N L D *  f_N L 1 -> f_N L D

            negative = torch.bmm(f_fake_part, f_real_part.permute(0, 2, 1)) / 0.5  # f_N L L
            positive1 = torch.bmm(f_real_part, f_real_part.permute(0, 2, 1)) / 0.5 # f_N L L
            l_neg = torch.sum(torch.exp(negative))

            l_pos1 = torch.sum(torch.exp(positive1))
            loss_real_intra = -torch.log(l_pos1 / (l_neg + l_pos1))
            #---------------fake----------------------
            positive2 = torch.bmm(f_fake_part, f_fake_part.permute(0, 2, 1)) / 0.5 # f_N L L
            l_pos2 = torch.sum(torch.exp(positive2))
            loss_fake_intra = -torch.log(l_pos2 / (l_neg + l_pos2))
            #loss_intra = loss_fake_intra + loss_real_intra
            loss_intra =  loss_real_intra

            real_index = (image_labels == 0).nonzero(as_tuple=True)[0]
            real_nums = len(real_index)
            if real_nums != 0 :
                real_embeddings = embeddings[real_index]  # N r_L D
                fake_nums = len(fake_index)
                if fake_nums >= real_nums:
                    random_fake_index = torch.randperm(fake_nums)[:real_nums]
                    random_fake_embeddings = fake_embeddings[random_fake_index]
                    real_neg = torch.bmm(real_embeddings, random_fake_embeddings.permute(0, 2, 1)) / 0.5
                    real_pos = torch.bmm(real_embeddings, real_embeddings.permute(0, 2, 1)) / 0.5

                else:
                    random_real_index = torch.randperm(real_nums)[:fake_nums]
                    random_real_embeddings = real_embeddings[random_real_index]
                    real_neg = torch.bmm(random_real_embeddings, fake_embeddings.permute(0, 2, 1)) / 0.5
                    real_pos = torch.bmm(random_real_embeddings, random_real_embeddings.permute(0, 2, 1)) / 0.5
                loss_real_neg = torch.sum(torch.exp(real_neg))
                loss_real_pos = torch.sum(torch.exp(real_pos))
                loss_inter = -torch.log(loss_real_pos / (loss_real_pos + loss_real_neg))
                loss_clip = loss_inter  + loss_intra

            else:
                loss_clip = loss_intra

        return loss_clip


class RecAttnClip(nn.Module):
    def __init__(self, vit, num_quires, device):
        super().__init__()
        self.vit = vit
        self.resblocks = self.vit.transformer.resblocks
        self.first_layer = 0
        self.clss_nums = num_quires
        self.ln_post = self.vit.ln_post
        self.proj = self.vit.proj
        self.num_features = self.vit.width
        self.device = device
        self.intra_scale = nn.Parameter(torch.zeros(1))
        self.intra_map = {6:0}
        self.clip_intra_blocks = nn.ModuleList([ClipIntraBlock(self.num_features).to(self.device) for _ in range(1)])
        self._freeze()

    def build_attn_mask(self, attn_bias):

        num_heads = self.resblocks[0].attn.num_heads
        n, Head, q, h, w = attn_bias.shape

        assert (
                Head == num_heads
        ), f"num_head={Head} is not supported. Modify to {num_heads}"
        attn_bias = attn_bias.reshape(n * Head, q, -1)
        l = attn_bias.shape[-1]
        attn_mask = attn_bias.new_zeros(q + 1 + l, q + 1 + l)
        attn_mask[:, :q] = -100
        attn_mask[torch.arange(q), torch.arange(q)] = 0
        attn_mask[:q, q] = -100
        attn_mask = attn_mask[None, ...].expand(
            n * Head, -1, -1
        ).clone()
        attn_mask[:, :q, -l:] = attn_bias
        # attn_mask (n*head,1+q+l,1+q+l)
        attn_biases = [attn_mask for _ in self.resblocks.children()]
        return attn_biases

    def _freeze(self):
        for name, param in self.named_parameters():
            if 'clip_intra_blocks' in name :
                param.requires_grad = True
            else:
                param.requires_grad = False


    def forward(self, data_dict, clip_features, attn_bias,inference=False, normalize=False):
        cls_token = clip_features[f'layer_{self.first_layer}_cls'].unsqueeze(1).permute(1, 0, 2).clone()  # ND->N1D->1ND
        vision_tokens = clip_features[self.first_layer].permute(1, 0, 2).clone()  # NLD->LND
        clss_token = cls_token.repeat(self.clss_nums, 1, 1)  # 1ND -> clss_nums, N,D

        x = torch.cat(
            [
                clss_token,
                cls_token,
                vision_tokens
            ],
            dim=0
        )  # (1+Q+L,N,D)
        x.requires_grad = True
        clip_L = vision_tokens.shape[0] #

        attn_biases = self.build_attn_mask(attn_bias)

        loss_clip = 0
        for i, blocks in enumerate(self.resblocks.children()):

            x = blocks(x, attn_biases[i])
            if i == 6:
                intra_x, loss_clip_tmp = self.clip_intra_blocks[self.intra_map[i]](x, data_dict, clip_L, inference)
                loss_clip = loss_clip_tmp + loss_clip
                x[-clip_L:, ...] = intra_x * 0.05 + x[-clip_L:, ...]

        x = x.permute(1, 0, 2)  # LND -> NLD
        clss_token = x[:, :self.clss_nums, :]
        clss_token = self.ln_post(clss_token)
        if self.proj is not None:
            clss_token = clss_token @ self.proj
        if normalize:
            clss_token = F.normalize(clss_token, dim=-1)

        return clss_token, loss_clip
