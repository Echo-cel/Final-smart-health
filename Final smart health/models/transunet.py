import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class PatchEmbed(nn.Module):
    """将图像分割成patches并进行线性嵌入"""
    def __init__(self, img_size=512, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x

class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    """多层感知机"""
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer块"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Transformer(nn.Module):
    """Transformer编码器"""
    def __init__(self, embed_dim, num_heads, num_layers, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

class DecoderBlock(nn.Module):
    """解码器块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class TransUNet(nn.Module):
    """TransUNet模型：结合Transformer和U-Net"""
    def __init__(self, img_size=512, in_channels=3, out_channels=1, 
                 embed_dim=768, patch_size=16, num_heads=12, num_layers=12, 
                 mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch Embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        
        # Position Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Transformer Encoder
        self.transformer = Transformer(embed_dim, num_heads, num_layers, mlp_ratio, dropout)
        
        # U-Net Decoder
        self.decoder_channels = [embed_dim, 256, 128, 64, 32]
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(len(self.decoder_channels) - 1):
            self.decoder_blocks.append(
                DecoderBlock(self.decoder_channels[i], self.decoder_channels[i + 1])
            )
        
        # Final output layer
        self.final_conv = nn.Conv2d(self.decoder_channels[-1], out_channels, kernel_size=1)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer encoding
        x = self.transformer(x)  # (B, n_patches, embed_dim)
        
        # Reshape for decoder
        B, n_patches, embed_dim = x.shape
        H = W = int(math.sqrt(n_patches))
        x = x.transpose(1, 2).reshape(B, embed_dim, H, W)
        
        # U-Net decoder with skip connections (simplified)
        skip_connections = []
        current = x
        
        for i, decoder_block in enumerate(self.decoder_blocks):
            if i < len(skip_connections):
                current = decoder_block(current, skip_connections[-(i+1)])
            else:
                # For simplicity, we'll use upsampling without skip connections
                current = decoder_block.up(current)
                current = decoder_block.conv[0](current)
                current = decoder_block.conv[1:](current)
        
        # Final output
        output = self.final_conv(current)
        
        return output

    def get_attention_maps(self, x):
        """获取注意力图用于可视化"""
        attention_maps = []
        
        # Patch embedding
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        # Get attention maps from each transformer block
        for block in self.transformer.blocks:
            # Get attention weights
            B, N, C = x.shape
            qkv = block.attn.qkv(block.norm1(x)).reshape(B, N, 3, block.attn.num_heads, block.attn.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            attn = (q @ k.transpose(-2, -1)) * (block.attn.head_dim ** -0.5)
            attn = F.softmax(attn, dim=-1)
            
            # Average attention across heads
            attn_map = attn.mean(dim=1)  # (B, N, N)
            attention_maps.append(attn_map)
            
            # Forward pass
            x = block(x)
        
        return attention_maps 