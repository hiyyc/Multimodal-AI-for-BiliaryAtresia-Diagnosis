import numpy as np
import torch
import torch.nn as nn
import compute, heads
from heads import MultiHeadAttention
from timm.models.layers import DropPath, trunc_normal_

from parts import build_image_encoder
from utils.tools import get_phase_list


class MultimodalCAFusion(nn.Module):
    def __init__(self, config, phase):
        super().__init__()
        self.config = config
        self.phase = phase
        self.img_types = config['img_types']
        self.tabular_len = config['tabular_len']

        self.dim_projection = config['dim_projection']
        self.image_encoder = build_image_encoder(self.config)

        if self.config['finetune']:
            for name, param in self.image_encoder.named_parameters():
                param.requires_grad = False

        hidden_size = self.config["hidden_size"]
        num_class = self.config["num_class"]
        if 'swin' in self.config['image_encoder_type']:
            self.image_projection = nn.Parameter(
                torch.empty(self.image_encoder.head.in_features, self.dim_projection),
            )
        else:
            self.image_projection = nn.Parameter(
                torch.empty(self.image_encoder.fc.in_features, self.dim_projection)
            )
            self.image_projection_from_feature = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
        trunc_normal_(self.image_projection, std=.02)

        self.pooler = heads.Pooler(hidden_size)
        self.pooler.apply(compute.init_weights)

        self.tabular_norm = nn.BatchNorm1d(self.tabular_len)

        self.classifier = nn.Sequential(
            nn.Linear(2048 + self.tabular_len, self.dim_projection),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.dim_projection, num_class),
        )
        self.classifier.apply(compute.init_weights)

        self.classifier_tabular = nn.Sequential(
            nn.Linear(self.tabular_len, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, num_class),
        )
        self.classifier_tabular.apply(compute.init_weights)

        self.classifier_img0 = nn.Sequential(
            nn.Linear(self.dim_projection, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, num_class),
        )
        self.classifier_img0.apply(compute.init_weights)

        self.classifier_img1 = nn.Sequential(
            nn.Linear(self.dim_projection, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, num_class),
        )
        self.classifier_img1.apply(compute.init_weights)

        # four losses
        self.classifier_img2 = nn.Sequential(
            nn.Linear(self.dim_projection, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, num_class),
        )
        self.classifier_img2.apply(compute.init_weights)

        self.classifier_img3 = nn.Sequential(
            nn.Linear(self.dim_projection, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, num_class),
        )
        self.classifier_img3.apply(compute.init_weights)

        self.multihead_attn = MultiHeadAttention(self.dim_projection, self.dim_projection, self.dim_projection,
                                                 num_hiddens=self.dim_projection, num_heads=4, dropout=0.1)
        self.criterion = nn.CrossEntropyLoss()

        self.train_phase = get_phase_list('train')
        self.test_phase = get_phase_list('test')

    def encode_image(self, image, norm=True):
        x = self.image_encoder.forward_features(image)
        if 'resnet' in self.config['image_encoder_type']:
            x = self.image_projection_from_feature(x)
        x = x @ self.image_projection
        if norm:
            x = x / x.norm(dim=-1, keepdim=True)
        if 'resnet' not in self.config['image_encoder_type']:
            x = x.mean(dim=1)
        return x

    def encode_tabular(self, tabular):
        tabular = tabular.repeat((1, 20))
        # tabular = self.tabular_norm(tabular)
        return tabular

    def infer(self, batch):
        """
        :param batch: {"record": [feature], "image": [img], "label": [label]}
        :return:
        """
        img_dict = {}
        for i in self.img_types:
            img_dict[f'image_embeds_{i}'] = self.encode_image(batch[f'image{i}'])
        tabular_embeds = self.encode_tabular(batch['record'])

        co_embed_list = [tabular_embeds]

        attn = self.config['attn']
        if attn == 'sa':
            k_v = []
            for i in self.img_types:
                k_v.append(torch.unsqueeze(img_dict[f'image_embeds_{i}'], 1))
            k_v_tensor = torch.cat(k_v, dim=1)
            for j in self.img_types:
                q = torch.unsqueeze(img_dict[f'image_embeds_{j}'], 1)
                img_dict[f'image_attn_{j}'] = self.multihead_attn(q, k_v_tensor, k_v_tensor, None)
                co_embed_list.append(torch.squeeze(img_dict[f'image_attn_{j}'], 1))
            # tmp_output = self.multihead_attn(k_v_tensor, k_v_tensor, k_v_tensor, None)
            # # co_embed_list.extend(tmp_output)
            # co_embeds = torch.cat([torch.unsqueeze(tabular_embeds, 1), tmp_output], dim=1)
        elif attn == 'sma':
            for i in self.img_types:
                k_v = []
                for j in self.img_types:
                    if i == j:
                        continue
                    k_v.append(torch.unsqueeze(img_dict[f'image_embeds_{j}'], 1))
                k_v_tensor = torch.cat(k_v, dim=1)
                q = torch.unsqueeze(img_dict[f'image_embeds_{i}'], 1)
                img_dict[f'image_attn_{i}'] = self.multihead_attn(q, k_v_tensor, k_v_tensor, None)
                co_embed_list.append(torch.squeeze(img_dict[f'image_attn_{i}'], 1))
        elif attn == 'none':
            for i in self.img_types:
                co_embed_list.append(img_dict[f'image_embeds_{i}'])

        co_embeds = torch.cat(co_embed_list, dim=1)

        return co_embeds, tabular_embeds, img_dict
        # return co_embeds

    def forward(self, batch):
        # test patient level
        if self.phase in self.test_phase:
            sample = {
                'label': int(batch['label']),
                'record': batch['record']
            }
            rlt = []
            for i in batch['image_combination']:
                sample.update(i)
                cls_logits = self.classifier(self.infer(sample)[0])
                # cls_logits = self.classifier(self.infer(sample))
                rlt.append(torch.squeeze(cls_logits))
            rlt_tensor = torch.tensor(np.array([item.cpu().detach().numpy() for item in rlt]))
            rlt_logits = torch.softmax(rlt_tensor, dim=1)
            rlt_mean = torch.mean(rlt_logits, dim=0)
            return rlt_mean
        else:
            co_embeds, tabular_embeds, img_dict = self.infer(batch)
            return torch.softmax(self.classifier(co_embeds), dim=1), \
                   torch.softmax(self.classifier_tabular(tabular_embeds), dim=1), \
                   torch.softmax(self.classifier_img0(img_dict['image_embeds_0']), dim=1), \
                   torch.softmax(self.classifier_img1(img_dict['image_embeds_1']), dim=1), \
                   torch.softmax(self.classifier_img2(img_dict['image_embeds_2']), dim=1), \
                   torch.softmax(self.classifier_img3(img_dict['image_embeds_3']), dim=1)
