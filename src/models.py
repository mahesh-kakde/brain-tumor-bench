# model definitions for all four architectures
import torch
import torch.nn as nn
from torchvision import models


def build_resnet50(num_classes=4, pretrained=True):
    """resnet-50 with custom 4-class head."""
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)
    # replace fc: dropout(0.2) -> linear(2048, 4)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(2048, num_classes),
    )
    return model


def build_efficientnet_b0(num_classes=4, pretrained=True):
    """efficientnet-b0 with custom 4-class head."""
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    # replace classifier: dropout(0.2) -> linear(1280, 4)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(1280, num_classes),
    )
    return model


def build_vit_b16(num_classes=4, pretrained=True):
    """vit-b/16 with custom 4-class head."""
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vit_b_16(weights=weights)
    # replace heads.head: dropout(0.2) -> linear(768, 4)
    model.heads.head = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(768, num_classes),
    )
    return model


def build_hybrid(num_classes=4, pretrained=True):
    """hybrid cnn-vit: resnet-50 backbone + 4-layer transformer head."""
    return HybridCNNViT(num_classes=num_classes, pretrained=pretrained)


class HybridCNNViT(nn.Module):
    """resnet-50 layer4 features -> transformer encoder -> 4-class output.

    data flow:
        input (b,3,224,224)
        -> resnet-50 layers 1-4  -> (b, 2048, 7, 7)
        -> flatten tokens        -> (b, 49, 2048)
        -> linear projection     -> (b, 49, 512)
        -> add pos embeddings    -> (b, 49, 512)
        -> prepend [cls] token   -> (b, 50, 512)
        -> transformer encoder   -> (b, 50, 512)
        -> [cls] output          -> (b, 512)
        -> dropout + linear      -> (b, 4)
    """

    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()

        # cnn backbone — resnet-50 up to layer4
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.cnn = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )  # output: (b, 2048, 7, 7)

        # projection: 2048 -> 512
        self.proj = nn.Linear(2048, 512)

        # learnable positional embeddings for 49 spatial tokens
        self.pos_embed = nn.Parameter(torch.randn(1, 49, 512) * 0.02)

        # learnable [cls] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512) * 0.02)

        # transformer encoder: 4 layers, 8 heads, dim=512, ffn=2048
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # classification head
        self.head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        b = x.size(0)

        # extract cnn features
        feat = self.cnn(x)                          # (b, 2048, 7, 7)
        feat = feat.flatten(2).transpose(1, 2)      # (b, 49, 2048)

        # project and add positional embeddings
        tokens = self.proj(feat) + self.pos_embed   # (b, 49, 512)

        # prepend [cls] token
        cls = self.cls_token.expand(b, -1, -1)      # (b, 1, 512)
        tokens = torch.cat([cls, tokens], dim=1)    # (b, 50, 512)

        # transformer
        out = self.transformer(tokens)              # (b, 50, 512)

        # classify from [cls] output
        return self.head(out[:, 0])                 # (b, num_classes)


# convenience registry
MODEL_BUILDERS = {
    "resnet50":        build_resnet50,
    "efficientnet_b0": build_efficientnet_b0,
    "vit_b16":         build_vit_b16,
    "hybrid":          build_hybrid,
}


def build_model(name, num_classes=4, pretrained=True):
    """build model by name. name must be one of MODEL_BUILDERS keys."""
    if name not in MODEL_BUILDERS:
        raise ValueError(f"unknown model '{name}'. choose from {list(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[name](num_classes=num_classes, pretrained=pretrained)