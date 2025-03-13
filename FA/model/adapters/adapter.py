import torch
from torch import nn
from timm import create_model
from torch.nn import functional as F

from model.layer import Fusion, MLP, PatchEmbed, VT_LN
from functools import partial


class Mask_Decoder(nn.Module):
    def __init__(self, in_dim, mlp_dim=512, out_dim=256, mlp_num_layers=3, head_num=16):
        super().__init__()
        self.in_dim = in_dim
        self.mlp_dim = mlp_dim
        self.out_dim = out_dim
        self.head_num = head_num
        dense_affine_func = partial(nn.Conv2d, kernel_size=1)
        self.query_mlp = MLP(in_dim, mlp_dim, out_dim, mlp_num_layers)  # L R L R L
        self.xray_mlp = MLP(in_dim, mlp_dim, out_dim, mlp_num_layers, affine_func=dense_affine_func)
        self.attn_mlp = MLP(in_dim, mlp_dim, out_dim * self.head_num, mlp_num_layers, affine_func=dense_affine_func)
        self.bias_scaling = nn.Linear(1, 1)

    def forward(self, query, x):
        # query (N,QL,D) x (N D H W)
        query = self.query_mlp(query)
        xray = self.xray_mlp(x)
        attn = self.attn_mlp(x)
        patch_x = x.reshape(x.shape[0], x.shape[1], -1) #(N D L)
        patch_x = patch_x.permute(0, 2, 1) #(N L D)
        xray_pred = torch.einsum('NQD,NDhw->NQhw', query, xray)
        n, d, h, w = xray.shape
        attn = attn.reshape(n, self.head_num, d, h, w)  # (N Head*D,h,w)->(N Head D h w)
        attn_bias = torch.einsum('NQD,NHDhw->NHQhw', query, attn)
        attn_bias = self.bias_scaling(attn_bias[..., None]).squeeze(-1)
        return xray_pred, attn_bias


class Adapter(nn.Module):
    def __init__(self, vit_name, num_quires, fusion_map, mlp_dim, mlp_out_dim, head_num, device):
        super().__init__()
        self.device = device
        self.vit_model = create_model(vit_name,
                                      pretrained=False,
                                      fc_norm=False,
                                      num_classes=0,
                                      embed_layer=PatchEmbed,
                                      ).to(device=self.device)

        if self.vit_model.cls_token is not None:
            self.vit_model.pos_embed = nn.Parameter(self.vit_model.pos_embed[:, 1:, ...])  # 去掉cls的位置
        del self.vit_model.cls_token
        self.vit_model.cls_token = None
        del self.vit_model.norm
        self.vit_model.norm = nn.Identity()
        self.num_quires = num_quires
        self.num_features = self.vit_model.num_features
        self.query_embed = nn.Parameter(torch.zeros(1, self.num_quires, self.num_features))  # (1,Q_L,D)
        self.query_pos_embed = nn.Parameter(torch.zeros(1, self.num_quires, self.num_features))
        self.fusion_map = fusion_map
        nn.init.normal_(self.query_embed, std=0.02)
        nn.init.normal_(self.query_pos_embed, std=0.02)
        self.mask_decoder = Mask_Decoder(in_dim=self.num_features, mlp_dim=mlp_dim, out_dim=mlp_out_dim,
                                         mlp_num_layers=3, head_num=head_num)
        self.ln_pre = VT_LN(self.num_features)
        self.patch_conv = nn.Conv2d(in_channels=3, out_channels=self.num_features, kernel_size=16, stride=16,
                                    bias=True)

    def fuse(self, block_idx, x, clip_features, spatial_shape):
        if block_idx in self.fusion_map.keys():
            clip_layer = self.fusion_map[block_idx]
            adapter_layer = block_idx
            clip_dim = clip_features[clip_layer].shape[2]  # clip features NLD

            fusion = Fusion(clip_dim, self.num_features).to(self.device)
            L = spatial_shape[0] * spatial_shape[1]
            x = torch.cat(
                [
                    x[:, :-L, ...],  # query
                    # fuse vision token x(N,a_L,D) clip_f[i] (N,c_L,D)
                    fusion(x[:, -L:, ...], clip_features[clip_layer], spatial_shape)
                ],
                dim=1
            )
            return x

    def intra_contra(self, block_idx, x, patch_labels, image_labels, spatial_shape):
        loss = 0
        if block_idx == 4 or block_idx == 5 or block_idx == 6 :
            L = spatial_shape[0] * spatial_shape[1]  # 14 * 14 = 196

            embeddings = x[:, -L:, ...]  # (N 196 192) (NLD)
            embeddings = nn.functional.normalize(embeddings, dim=-1)
            fake_index = (image_labels == 1).nonzero(as_tuple=True)[0]
            fake_embeddings = embeddings[fake_index]  # f_N L D
            fake_nums = len(fake_index)

            ff_patch_labels = patch_labels[fake_index]  # f_N L


            fr_patch_labels = torch.logical_not(ff_patch_labels)

            f_fake_part = fake_embeddings * ff_patch_labels.unsqueeze(-1)  # f_N L D *  f_N L 1 -> f_N L D
            f_real_part = fake_embeddings * fr_patch_labels.unsqueeze(-1)  # f_N L D *  f_N L 1 -> f_N L D

            negative = torch.bmm(f_fake_part, f_real_part.permute(0, 2, 1)) / 0.5  # f_N L L
            positive1 = torch.bmm(f_real_part, f_real_part.permute(0, 2, 1)) / 0.5  # f_N L L
            positive2 = torch.bmm(f_fake_part, f_fake_part.permute(0, 2, 1)) / 0.5  # f_N L L

            l_neg = torch.sum(torch.exp(negative))
            l_pos1 = torch.sum(torch.exp(positive1))
            l_pos2 = torch.sum(torch.exp(positive2))

            loss_real_intra = -torch.log(l_pos1 / (l_neg + l_pos1))
            loss_fake_intra = -torch.log(l_pos2 / (l_neg + l_pos2))


            #loss = loss_real_intra + loss_fake_intra
            loss = loss_real_intra

        return loss

    def forward(self, data_dict, clip_features, inference):

        image = data_dict['image']
        x = self.patch_conv(image)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        pos_embed = self.vit_model.pos_embed  # (N L D)
        pos_embed = pos_embed.permute(0, 2, 1) #(NDL)
        pos_embed = F.interpolate(pos_embed.reshape(pos_embed.shape[0],pos_embed.shape[1], 14, 14),
                                  size=(16, 16),
                                  mode='bilinear',
                                  align_corners=False).reshape(pos_embed.shape[0],pos_embed.shape[1],256).permute(0,2,1) #NDL->NLD
        pos_embed = torch.cat([self.query_pos_embed.expand(pos_embed.shape[0], -1, -1), pos_embed],
                              dim=1)
        v_L = x.shape[1]  # vision token L 196
        (h, w) = 16,16  # h w 16,16

        x = torch.cat([self.query_embed.expand(x.shape[0], -1, -1), x], dim=1)  # (N ,Q_L+L,D)
        x = x + pos_embed
        x = self.ln_pre(x)
        outs = []
        out_layers = [8]
        loss_intra = 0
        # self.fuse(0, x, clip_features, (h, w))
        for i, block in enumerate(self.vit_model.blocks, start=1):  # total 1-12 ,only use 1-8
            x = block(x)  # (N, Q_L+L, D)
            self.fuse(i, x, clip_features, (h, w))
            if not inference:  #train
                loss_tmp_intra = self.intra_contra(i, x, data_dict['patch_label'], data_dict['label'], (h, w))
                loss_intra = loss_tmp_intra + loss_intra if loss_tmp_intra != 0 else loss_intra
            if i in out_layers:
                n, _, d = x.shape
                outs.append(
                    {
                        'query': x[:, :-v_L, ...],
                        'x': x[:, -v_L:, ...]
                        .permute(0, 2, 1)
                        .reshape(n, d, h, w),
                    }
                )
            x = x + pos_embed
            if i == max(out_layers):
                break
        xray_preds = []
        attn_biases = []

        for feature in outs:
            xray_pred, attn_bias = self.mask_decoder(feature['query'], feature['x'])
            xray_preds.append(xray_pred)
            attn_biases.append(attn_bias)

        return attn_biases, xray_preds, loss_intra
