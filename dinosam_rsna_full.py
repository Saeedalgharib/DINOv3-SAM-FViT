import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import SamModel


class Config:
    dino_model = "vit_small_patch16_dinov3"
    sam_model = "facebook/sam-vit-base"

    dino_dim = 384
    sam_dim = 256
    fusion_dim = dino_dim + sam_dim

    # ============================================================
    # x = number of output classes
    # Example:
    #   x = 2 → binary classification (RSNA, Mooney, etc.)
    #   x = 3 → COVID (3 classes)
    #   x = 7 → Dermatology MNIST (7 classes)
    # ============================================================
    x = 7 
    num_classes = x
    classes = list(range(x))


class DINOv3FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "vit_small_patch16_dinov3",
            pretrained=True,
            num_classes=0
        )

        for p in self.model.parameters():
            p.requires_grad = False

        for block in self.model.blocks[-3:]:
            for p in block.parameters():
                p.requires_grad = True

    def forward(self, x):
        tokens = self.model.forward_features(x)

        if isinstance(tokens, torch.Tensor):
            if tokens.dim() == 4:
                return tokens
            elif tokens.dim() == 3:
                B, N, C = tokens.shape

                if N == 197:
                    patch_tokens = tokens[:, 1:, :]
                else:
                    patch_tokens = tokens

                N_p = patch_tokens.shape[1]
                H = W = int(math.sqrt(N_p))

                patch_tokens = patch_tokens[:, :H * W, :]
                fm = patch_tokens.view(B, H, W, C).permute(0, 3, 1, 2)

                return fm
            else:
                raise RuntimeError(f"Unexpected DINO tokens shape: {tokens.shape}")
        else:
            raise RuntimeError("DINO forward_features did not return a Tensor.")


class SAMFeatureExtractor(nn.Module):
    def __init__(self, model_name="facebook/sam-vit-base"):
        super().__init__()
        self.model = SamModel.from_pretrained(model_name)

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x):
        vision_out = self.model.vision_encoder(pixel_values=x)
        feats = vision_out.last_hidden_state

        if feats.dim() == 4:
            fm = F.adaptive_avg_pool2d(feats, (14, 14))
            return fm
        else:
            raise RuntimeError(f"Unexpected SAM features shape: {feats.shape}")


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )

        self.spatial_conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=7,
            padding=3
        )

    def forward(self, x):
        B, C, H, W = x.shape

        avg_pool = F.adaptive_avg_pool2d(x, 1).view(B, C)
        max_pool = F.adaptive_max_pool2d(x, 1).view(B, C)

        mlp_avg = self.mlp(avg_pool)
        mlp_max = self.mlp(max_pool)

        channel_att = torch.sigmoid(mlp_avg + mlp_max).view(B, C, 1, 1)
        x = x * channel_att

        avg_sp = torch.mean(x, dim=1, keepdim=True)
        max_sp, _ = torch.max(x, dim=1, keepdim=True)
        sp_cat = torch.cat([avg_sp, max_sp], dim=1)

        spatial_att = torch.sigmoid(self.spatial_conv(sp_cat))
        x = x * spatial_att

        return x


class DinoSamFusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dino = DINOv3FeatureExtractor()
        self.sam = SAMFeatureExtractor(config.sam_model)

        self.cbam = CBAM(channels=config.fusion_dim)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.head_fused = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(config.fusion_dim, config.num_classes)
        )

        self.head_dino = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(config.dino_dim, config.num_classes)
        )

        self.head_sam = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(config.sam_dim, config.num_classes)
        )

    def forward(self, dino_img, sam_img):
        fm_dino = self.dino(dino_img)
        fm_sam = self.sam(sam_img)

        fused = torch.cat([fm_dino, fm_sam], dim=1)
        fused = self.cbam(fused)

        dino_vec = self.gap(fm_dino).view(fm_dino.size(0), -1)
        sam_vec = self.gap(fm_sam).view(fm_sam.size(0), -1)
        fused_vec = self.gap(fused).view(fused.size(0), -1)

        dino_logits = self.head_dino(dino_vec)
        sam_logits = self.head_sam(sam_vec)
        fused_logits = self.head_fused(fused_vec)

        return fused_logits, dino_logits, sam_logits


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()

        p_t = (probs * one_hot).sum(dim=1)
        log_p_t = (log_probs * one_hot).sum(dim=1)

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
        else:
            alpha_t = 1.0

        loss = - alpha_t * (1.0 - p_t) ** self.gamma * log_p_t

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
