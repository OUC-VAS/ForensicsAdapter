import numpy as np
from sklearn import metrics
from torch import nn
from model.clip.clip import load
import torch
from model.adapters.adapter import Adapter
from .attn import RecAttnClip
from .layer import PostClipProcess, MaskPostXrayProcess
import torch.nn.functional as F
from trainer.metrics.base_metrics_class import calculate_metrics_for_train


class DS(nn.Module):
    def __init__(self, clip_name,
                 adapter_vit_name,
                 num_quires,
                 fusion_map,
                 mlp_dim,
                 mlp_out_dim,
                 head_num,
                 device,
                 mode='video'):
        super().__init__()
        self.device = device
        self.clip_model, self.processor = load(clip_name, device=device,download_root='/data/cuixinjie/weights')
        self.adapter = Adapter(vit_name=adapter_vit_name, num_quires=num_quires, fusion_map=fusion_map, mlp_dim=mlp_dim,
                               mlp_out_dim=mlp_out_dim, head_num=head_num, device=self.device)
        self.rec_attn_clip = RecAttnClip(self.clip_model.visual, num_quires,device=self.device)  # 全部参数被冻结
        self.masked_xray_post_process = MaskPostXrayProcess(in_c=num_quires).to(self.device)
        self.clip_post_process = PostClipProcess(num_quires=num_quires, embed_dim=768)

        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        self.mode = mode
        self._freeze()


    def _freeze(self):
        for name, param in self.named_parameters():
            if 'clip_model' in name :
                param.requires_grad = False

    def get_losses(self, data_dict, pred_dict):
        label = data_dict['label'] #N
        xray = data_dict['xray']
        pred = pred_dict['cls']  #N2
        xray_pred = pred_dict['xray_pred']
        loss_intra = pred_dict['loss_intra']
        loss_clip = pred_dict['loss_clip']
        criterion = nn.CrossEntropyLoss()
        loss1 = criterion(pred.float(), label)
        if xray is not None:
            loss_mse = F.mse_loss(xray_pred.squeeze().float(), xray.squeeze().float())  # (N 1 224 224)->(N 224 224)

            loss = 10 * loss1 + 200 * loss_mse + 20 * loss_intra + 10 * loss_clip


            loss_dict = {
                'cls': loss1,
                'xray': loss_mse,
                'intra': loss_intra,
                'loss_clip':loss_clip,
                'overall': loss
            }
            return loss_dict
        else:
            loss_dict = {
                'cls': loss1,
                'overall': loss1
            }
            return loss_dict

    def get_train_metrics(self, data_dict, pred_dict):
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def get_test_metrics(self):
        y_pred = np.concatenate(self.prob)
        y_true = np.concatenate(self.label)
        # auc
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        # eer
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        # ap
        ap = metrics.average_precision_score(y_true, y_pred)
        # acc
        acc = self.correct / self.total
        # reset the prob and label
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'pred': y_pred, 'label': y_true}

    def forward(self, data_dict, inference=False):
        images = data_dict['image']
        clip_images = F.interpolate(
            images,
            size=(224, 224),
            mode='bilinear',
            align_corners=False,
        )


        clip_features = self.clip_model.extract_features(clip_images, self.adapter.fusion_map.values())

        attn_biases, xray_preds, loss_adapter_intra = self.adapter(data_dict, clip_features,
                                                                                inference)
        clip_output, loss_clip = self.rec_attn_clip(data_dict, clip_features, attn_biases[-1], inference, normalize=True)

        data_dict['if_boundary'] = data_dict['if_boundary'].to(self.device)
        xray_preds = [self.masked_xray_post_process(xray_pred, data_dict['if_boundary']) for xray_pred in xray_preds]

        clip_cls_output = self.clip_post_process(clip_output.float()).squeeze()   # N2

        outputs = {
            'xray_pred': xray_preds[-1],  # N 1 224 224
            'clip_cls_output': clip_cls_output,  # N 2

        }

        prob = torch.softmax(outputs['clip_cls_output'], dim=1)[:, 1]
        pred_dict = {
            'cls': outputs['clip_cls_output'],
            'prob': prob,
            'xray_pred': outputs['xray_pred'],
            'loss_intra': loss_adapter_intra,
            'loss_clip':loss_clip,
        }

        if inference:
            self.prob.append(
                pred_dict['prob']
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            self.label.append(
                data_dict['label']
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            # deal with acc
            _, prediction_class = torch.max(outputs['clip_cls_output'], 1)
            correct = (prediction_class == data_dict['label']).sum().item()
            self.correct += correct
            self.total += data_dict['label'].size(0)

        return pred_dict
