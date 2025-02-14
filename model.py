# Defining the model
import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, num_heads, num_layers, num_classes):
        super(VisionTransformer, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.patch_dim = 3 * patch_size * patch_size  # 3 color channels
        
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim*4), num_layers
        )
        self.mlp_head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        B, _, _, _ = x.shape # Get batch size. We don't need the channels, height, or width
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, self.num_patches, -1)
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed

        x = self.transformer(x)
        x = x[:, 0, :]
        return self.mlp_head(x)

def build_transformer(config):
    return VisionTransformer(
        config['img_size'], config['patch_size'], config['embed_dim'], config['num_heads'],
        config['num_layers'], config['num_classes']
    )