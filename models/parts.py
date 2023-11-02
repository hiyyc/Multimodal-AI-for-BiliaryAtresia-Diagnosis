import numpy as np
import torch
import torch.nn as nn
from timm.models import create_model


def build_image_encoder(config, num_classes=2):
    model_type = config['image_encoder_type']
    print(f"Creating model: {model_type}")
    is_pretrained = True
    # is_pretrained = False

    if "swin" in model_type:
        model = create_model(
            model_type,
            pretrained=is_pretrained,
            img_size=config['img_size'],
            num_classes=num_classes,
        )
    elif "vit" in model_type:
        model = create_model(
            model_type,
            pretrained=is_pretrained,
            img_size=config.DATA.IMG_SIZE,
            num_classes=num_classes,
        )
    elif "resnet" in model_type:
        model = create_model(
            model_type,
            pretrained=is_pretrained,
            num_classes=num_classes
        )
    else:
        model = create_model(
            model_type,
            pretrained=is_pretrained,
            num_classes=num_classes
        )
    return model


class TabularEmbeddings(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super().__init__()
        self.feature_size = feature_size
        self.layer = nn.Linear(self.feature_size, hidden_size)

    def forward(self, x):
        return self.layer(x[:, :self.feature_size]).unsqueeze(-2), torch.full((x.shape[0], 1), 1, device=x.device)


class CNNFusion(nn.Module):
    def __init__(self, in_f, out_f):
        super(CNNFusion, self).__init__()
        self.in_f = in_f
        self.out_f = out_f
        self.layer = nn.Sequence(
            nn.Linear(in_f, 1024),
            nn.Relu(out_f),
            nn.Linear(1024, out_f)
        )

    def forward(self, x):
        return self.layer(x)


def rollout(attentions, discard_ratio, head_fusion):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1, keepdim=True)

            if a.shape[0] == 16:
                a = a.repeat(4, 1, 1)
            elif a.shape[0] == 4:
                a = a.repeat(16, 1, 1)
            result = torch.matmul(a, result)

    mask = result[0, 0, :]
    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask


class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean",
                 discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)

        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.detach().cpu())

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)
        return rollout(self.attentions, self.discard_ratio, self.head_fusion)


torch.autograd.set_detect_anomaly(True)


class TransformerAttentionRollout:
    def __init__(self, model):
        self.model = model
        for name, module in self.model.named_modules():
            if 'attn' in name:
                module.register_forward_hook(self.get_attention)
                module.register_backward_hook(self.get_grad)
        self.attentions = []
        self.grads = []

        self.criterion = nn.CrossEntropyLoss()

    def get_attention(self, module, input, output):
        self.attentions.append(output.detach().cpu())

    def get_grad(self, module, input, output):
        self.grads.append(output)

    def __call__(self, input_tensor, index):
        self.attentions = []
        self.grads = []
        # with torch.no_grad():
        output = self.model(input_tensor)

        one_hot = torch.zeros((1, output.size()[-1]), requires_grad=True).cuda()
        one_hot[0][index] = 1
        one_hot = torch.sum(one_hot * output)
        self.model.zero_grad()
        one_hot.backward()

        for k, v in self.model.named_parameters():
            if v.grad is None:
                continue
            self.grads.append(v.grad)
        grad = self.grads[-1]
        cam = self.attentions[-1]
        cam = cam[0, 0, 180:].reshape(-1, 14, 14).cuda()
        grad = grad[180:].reshape(-1, 14, 14).cuda()
        grad = grad.mean(dim=[1, 2], keepdim=True)
        cam = (cam * grad).mean(0).clamp(min=0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam
