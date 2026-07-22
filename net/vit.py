"""Vision Transformer models adapted to the DDU feature interface."""

from torch import nn
from timm.models.vision_transformer import VisionTransformer


class CIFARViT(nn.Module):
    """ViT-Tiny/4 for 32x32 CIFAR images trained from scratch."""

    def __init__(
        self,
        spectral_normalization=False,
        mod=False,
        coeff=3.0,
        num_classes=10,
        mnist=False,
        temp=1.0,
    ):
        super().__init__()
        if spectral_normalization:
            raise ValueError("vit_tiny_patch4_32 does not support spectral normalization")
        if mod:
            raise ValueError("vit_tiny_patch4_32 does not support CNN architectural modifications")
        if mnist:
            raise ValueError("vit_tiny_patch4_32 only supports 3-channel 32x32 inputs")

        # coeff is accepted to preserve the common model factory signature.
        del coeff

        self.feature_dim = 192
        self.backbone = VisionTransformer(
            img_size=32,
            patch_size=4,
            in_chans=3,
            num_classes=0,
            global_pool="token",
            embed_dim=self.feature_dim,
            depth=12,
            num_heads=3,
            mlp_ratio=4.0,
            qkv_bias=True,
            class_token=True,
            final_norm=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
        )
        self.fc = nn.Linear(self.feature_dim, num_classes)
        self.feature = None
        self.temp = temp

    def forward(self, x):
        tokens = self.backbone.forward_features(x)
        features = self.backbone.forward_head(tokens, pre_logits=True)
        self.feature = features.detach()
        return self.fc(features) / self.temp


def vit_tiny_patch4_32(**kwargs):
    return CIFARViT(**kwargs)
