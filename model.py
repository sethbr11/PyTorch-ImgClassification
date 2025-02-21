# Defining the model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, num_heads, num_layers, num_classes, dropout_rate):
        super(VisionTransformer, self).__init__()

        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Conv2d-based Patch Embedding (Better Stability)
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Class Token (Learnable) & Positional Embedding (Learnable)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Transformer with Pre-Normalization (More Stable Training)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim*4, 
                                                   dropout=dropout_rate, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # LayerNorm after each transformer block for stability
        self.norm = nn.LayerNorm(embed_dim)

        # Dropout and Classification Head
        self.dropout = nn.Dropout(dropout_rate)
        self.mlp_head = nn.Linear(embed_dim, num_classes)

        # Initialize Weights
        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_mlp_weights)

    def _init_mlp_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]

        # Patch Embedding (Conv2d)
        x = self.patch_embed(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

        # Class Token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches+1, embed_dim]

        # Positional Encoding (Learnable)
        x = x + self.pos_embed[:, :x.shape[1], :]

        # Transformer Encoder with Pre-Normalization
        x = self.transformer(x)

        # Apply LayerNorm after transformer blocks
        x = self.norm(x)

        # Use Global Average Pooling (GAP) for classification
        x = x.mean(dim=1)  # Apply Global Average Pooling (GAP) across the sequence
        x = self.dropout(x)  # Apply dropout before the final layer
        return self.mlp_head(x)

def build_transformer(config):
    return VisionTransformer(
        config['img_size'], config['patch_size'], config['embed_dim'], config['num_heads'],
        config['num_layers'], config['num_classes'], config['dropout_rate']
    )
    
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
def build_cnn(config):
    return SimpleCNN(num_classes=config['num_classes'])