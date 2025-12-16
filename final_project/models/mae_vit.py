"""
Masked Autoencoder (MAE) with Vision Transformer Tiny backbone.

This implementation follows the MAE paper (He et al., CVPR 2022) with
ViT-Tiny architecture for single-GPU training constraints.

Architecture specs (ViT-Tiny):
- Patch size: 16x16
- Embed dim: 192
- Depth: 12 layers
- Num heads: 3
- MLP ratio: 4
- Decoder: 4 layers, 512 dim, 8 heads
"""

import torch
import torch.nn as nn
from einops import rearrange, repeat
from typing import Optional, Tuple


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.

    Converts images into flattened patches using a convolutional layer.
    This is more efficient than manually reshaping for large images.
    """

    def __init__(self, img_size: int = 64, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size

        # Convolutional projection: treats each patch as a kernel
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input images
        Returns:
            (B, num_patches, embed_dim) patch embeddings
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})"

        # Conv2d output: (B, embed_dim, grid_size, grid_size)
        x = self.proj(x)
        # Flatten spatial dimensions: (B, embed_dim, num_patches)
        x = x.flatten(2)
        # Transpose to (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        return x


class Attention(nn.Module):
    """
    Multi-head Self Attention with optional masking support.

    Standard transformer attention used in both encoder and decoder.
    """

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV projection in one matrix for efficiency
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) input features
        Returns:
            (B, N, C) attention output
        """
        B, N, C = x.shape

        # Generate Q, K, V: (B, N, 3, num_heads, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, N, head_dim)

        # Attention: softmax(Q @ K^T / sqrt(d)) @ V
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """
    MLP block with GELU activation.

    Standard transformer FFN: Linear -> GELU -> Dropout -> Linear -> Dropout
    """

    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block: LayerNorm -> Attention -> LayerNorm -> MLP.

    Uses pre-normalization (LayerNorm before attention/MLP) which is
    more stable for deep networks.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4., qkv_bias: bool = True, drop: float = 0., attn_drop: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual connections
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MAE_ViT(nn.Module):
    """
    Masked Autoencoder with Vision Transformer Tiny backbone.

    Architecture:
    1. Patch embedding
    2. Random/structured masking
    3. ViT encoder (only on visible patches for efficiency)
    4. Lightweight decoder (on all patches)
    5. Pixel reconstruction loss

    Args:
        img_size: Input image size (default: 64 for TinyImageNet)
        patch_size: Patch size (default: 16)
        in_chans: Number of input channels (default: 3)
        embed_dim: Encoder embedding dimension (default: 192 for ViT-Tiny)
        depth: Number of encoder blocks (default: 12 for ViT-Tiny)
        num_heads: Number of attention heads (default: 3 for ViT-Tiny)
        decoder_embed_dim: Decoder embedding dimension (default: 512)
        decoder_depth: Number of decoder blocks (default: 4)
        decoder_num_heads: Number of decoder attention heads (default: 8)
        mlp_ratio: MLP hidden dim ratio (default: 4)
        norm_pix_loss: Normalize pixels for loss calculation (default: True)
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 4,
        decoder_num_heads: int = 8,
        mlp_ratio: float = 4.,
        norm_pix_loss: bool = True,
    ):
        super().__init__()

        # Encoder
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token not used in MAE (unlike standard ViT)
        # Only need position embeddings for patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # Reconstruction head: project to pixel space
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans)

        self.norm_pix_loss = norm_pix_loss
        self.patch_size = patch_size
        self.num_patches = num_patches

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights following MAE paper."""
        # Position embeddings: Xavier uniform
        torch.nn.init.xavier_uniform_(self.pos_embed)
        torch.nn.init.xavier_uniform_(self.decoder_pos_embed)

        # Mask token: Xavier normal
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # Linear layers: Xavier uniform, biases to zero
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patches for loss calculation.

        Args:
            imgs: (B, C, H, W)
        Returns:
            (B, num_patches, patch_size^2 * C)
        """
        p = self.patch_size
        B, C, H, W = imgs.shape
        h = w = H // p

        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (B, h, w, p, p, C)
        x = x.reshape(B, h * w, p * p * C)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patches back to images for visualization.

        Args:
            x: (B, num_patches, patch_size^2 * C)
        Returns:
            (B, C, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)

        x = x.reshape(x.shape[0], h, w, p, p, 3)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (B, C, h, p, w, p)
        imgs = x.reshape(x.shape[0], 3, h * p, w * p)
        return imgs

    def forward_encoder(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder with masking.

        Key efficiency: only process visible (unmasked) patches.

        Args:
            x: (B, C, H, W) input images
            mask: (B, num_patches) binary mask (1 = keep, 0 = remove)
        Returns:
            latent: (B, num_visible, embed_dim) encoded visible patches
            mask: (B, num_patches) the mask
            ids_restore: (B, num_patches) indices to restore original order
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add position embeddings
        x = x + self.pos_embed

        # Create ids_restore for later unshuffling
        B, N, D = x.shape
        ids_shuffle = torch.argsort(mask, dim=1, descending=True)  # Visible patches first
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep only visible patches
        num_visible = mask.sum(dim=1)[0].int().item()  # Assume same for all in batch
        ids_keep = ids_shuffle[:, :num_visible]
        x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # Apply transformer blocks
        for blk in self.blocks:
            x_visible = blk(x_visible)
        x_visible = self.norm(x_visible)

        return x_visible, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.

        Args:
            x: (B, num_visible, embed_dim) encoded visible patches
            ids_restore: (B, num_patches) indices to restore patch order
        Returns:
            (B, num_patches, patch_size^2 * C) reconstructed patches
        """
        # Embed to decoder dimension
        x = self.decoder_embed(x)

        # Append mask tokens
        B, num_visible, D = x.shape
        mask_tokens = self.mask_token.repeat(B, self.num_patches - num_visible, 1)
        x_full = torch.cat([x, mask_tokens], dim=1)  # (B, num_patches, decoder_embed_dim)

        # Unshuffle to restore original order
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D))

        # Add position embeddings
        x_full = x_full + self.decoder_pos_embed

        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x_full = blk(x_full)
        x_full = self.decoder_norm(x_full)

        # Project to pixels
        x_full = self.decoder_pred(x_full)  # (B, num_patches, patch_size^2 * C)

        return x_full

    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Calculate reconstruction loss on masked patches only.

        Args:
            imgs: (B, C, H, W) original images
            pred: (B, num_patches, patch_size^2 * C) predicted patches
            mask: (B, num_patches) binary mask (1 = keep, 0 = masked)
        Returns:
            Scalar loss value
        """
        target = self.patchify(imgs)  # (B, num_patches, patch_size^2 * C)

        if self.norm_pix_loss:
            # Normalize target per patch (improves representation quality)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5

        # MSE loss only on masked patches
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Mean per patch

        # Calculate mean loss only on removed patches
        mask_binary = (mask == 0).float()  # 1 for masked, 0 for visible
        loss = (loss * mask_binary).sum() / mask_binary.sum()

        return loss

    def forward(self, imgs: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            imgs: (B, C, H, W) input images
            mask: (B, num_patches) binary mask (1 = keep, 0 = remove)
        Returns:
            loss: Reconstruction loss
            pred: Reconstructed patches
            mask: The mask used
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, mask)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Encode images without masking (for downstream tasks).

        Args:
            imgs: (B, C, H, W) input images
        Returns:
            (B, num_patches, embed_dim) patch representations
        """
        x = self.patch_embed(imgs)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x


def mae_vit_tiny(**kwargs):
    """Factory function for MAE with ViT-Tiny backbone."""
    model = MAE_ViT(
        embed_dim=192,
        depth=12,
        num_heads=3,
        decoder_embed_dim=512,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4,
        **kwargs
    )
    return model
