"""
任务: 使用timm库完成对RetrievalViT模型的建构
时间: 2024/10/17-Redal
"""
import os
import torch
import numpy as np
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import Block
from timm.models.layers import to_2tuple
from einops.layers.torch import Rearrange
from einops import rearrange,repeat



def get_1d_sincos_pos_embed_from_grid(embed_dim,pos):
    """
    embed_dim: 输出output的每个位置的输出尺寸
    pos: 要编码的位置列表: (M,)
    output: (M, D)
    """
    assert embed_dim % 2==0 , '====warning: Embed_dim must be divisible by 2'
    omega = np.arange(embed_dim//2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega 

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega) #计算外积: (M, D/2)
    embed_sin = np.sin(out)
    embed_cos = np.cos(out)
    embed_output = np.concatenate([embed_cos,embed_sin], axis=1) #(M, D)
    return embed_output

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    embed_dim: 输出output的每个位置的输出尺寸
    grid: 图像高和宽,要编码的位置列表: (H, W)
    output: (H*W, D)
    """
    assert embed_dim % 2==0 ,'====warning: Embed_dim must be divisible by 2'
    embed_dim_h = get_1d_sincos_pos_embed_from_grid(embed_dim//2, grid[0]) #(H, D/2)
    embed_dim_w = get_1d_sincos_pos_embed_from_grid(embed_dim//2, grid[1]) #(W, D/2)
    embed_output = np.concatenate([embed_dim_h,embed_dim_w], axis=1) #(H*W, D)
    return embed_output

def get_3d_sincos_pos_embed_from_grid(embed_dim, grid_size, cls_token=False):
    """
    grid_size: 网格大小的 3d 元组: (T, H, W)
    output: pos_embed: L, D
    """
    assert embed_dim % 16 == 0 , '====warning: Embed_dim must be divisible by 16'
    t_size, h_size, w_size = grid_size

    w_embed_dim = embed_dim // 16 * 6
    h_embed_dim = embed_dim // 16 * 6
    t_embed_dim = embed_dim // 16 * 4

    w_pos_embed = get_1d_sincos_pos_embed_from_grid(w_embed_dim, np.arange(w_size)) # (W, 6/16D)
    h_pos_embed = get_1d_sincos_pos_embed_from_grid(h_embed_dim, np.arange(h_size)) # (H, 6/16D)
    t_pos_embed = get_1d_sincos_pos_embed_from_grid(t_embed_dim, np.arange(t_size)) # (T, 4/16D)

    w_pos_embed = np.tile(w_pos_embed, (t_size * h_size, 1))                        # (W*T*H, 6/16D)
    h_pos_embed = np.tile(np.repeat(h_pos_embed, w_size, axis=0), (t_size, 1))      # (H*W*T, 6/16D) 三者均保持(embed_dim, Di)
    t_pos_embed = np.repeat(t_pos_embed, h_size * w_size, axis=0)                   # (T*W*H, 4/16D)

    pos_embed = np.concatenate((w_pos_embed, h_pos_embed, t_pos_embed), axis=1)
    if cls_token:
        pos_embed = np.concatenate((np.zeros([1,embed_dim]), pos_embed), axis=0)
    return pos_embed
    

class PatchEmbed(nn.Module):
    """
    输入图像分割成小块patches,再嵌入到高维空间中
    """
    def __init__(self,img_size = 224,
                 patch_size = 16,
                 num_frames=3,
                 tubelet_size=1,
                 in_chans = 3,
                 embed_dim = 768,
                 norm_layer = None,
                 flatten = True,
                 bias = True,):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size 
        self.grid_size = (num_frames // tubelet_size, img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten
        self.proj = nn.Conv3d(in_chans, embed_dim, 
                            kernel_size=(tubelet_size, patch_size[0],patch_size[1]),
                            stride=(tubelet_size, patch_size[0],patch_size[1]), bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self,x):
        B, C, T, H, W = x.shape
        x = self.proj(x)
        if self.flatten: 
            x = x.flatten(2).transpose(1, 2) # B,C,T,H,W -> B,C,L -> B,L,C
        x = self.norm(x)
        return x

class RetrievalViT(nn.Module):
    """
    带有 VisionTransform 主干的编码器
    """
    def __init__(self,img_size = 224, patch_size = 16,
                    num_frames = 3, tubelet_size=1,
                    in_chans = 3, embed_dim = 1024,
                    depth = 24, num_heads = 8, mlp_ratio = 4.,
                    norm_layer = partial(torch.nn.LayerNorm, eps=1e-6),
                    norm_pix_loss = False, weights = None, *args, **kwargs):
        super().__init__()

        # MAE 编码器细节 MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, num_frames, tubelet_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # (1, D)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad= False) # 需要加上'1'作分类标记头: (num_patches + 1, D)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) # 层归一化


        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

        # 负载模型权重load model weights
        if os.path.isfile(weights):
            state_dict = torch.load(weights, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            del state_dict['pos_embed'] # 丢弃pos_embedding权重
            self.load_state_dict(state_dict, strict=False)
        else:
            print(f'==== warning: {weights} not exists')
    
    def initialize_weights(self):
        # 通过 sin-cos 嵌入初始化（和冻结）pos_embed
        pos_embed = get_3d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # 使用以下 xavier_uniform 官方 JAX ViT
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: B,C,T,H,W
        x: B,L,D
        """
        p = self.patch_embed.patch_size[0]
        tub = self.patch_embed.tubelet_size
        x = rearrange(imgs, 'b c (t tub) (h p) (w q) -> b (t h w) (tub p q c)', tub=tub, p=p, q=p)
        return x

    def unpatchify(self, x):
        """
        x: B, L, D
        imgs: B, C, T, H, W
        """
        p = self.patch_embed.patch_size[0]
        num_p = self.patch_embed.img_size[0] // p
        tub = self.patch_embed.tubelet_size
        imgs = rearrange(x, 'b (t h w) (tub p q c) -> b c (t tub) (h p) (w q)', h=num_p, w=num_p, tub=tub, p=p, q=p)
        return imgs 

    def forward(self, x):
        x = self.patch_embed(x)
        # 添加 pos 嵌入 w/o cls 令牌
        x = x + self.pos_embed[:, 1:, :] # (B, num_patches + 1, D)

        cls_token = self.cls_token +self.pos_embed[:, :1, :]
        cls_token = cls_token.expand(x.shape[0], -1, -1)
        x =torch.cat((cls_token, x), dim=1)

        # 应用transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x[:, 1:].mean(dim=1)
        return x


if __name__ == "__main__":
    # RetrievalViT 模型实例化测试
    # Prithvi_100M.pt模型图像输入通道in_chans=6,后续网络添加需要增加图像通道
    Prithvi_100M_filepath = '/home3/dataset/tianzhibei/retrieval-main/Prithvi_100M.pt'
    vit_model = RetrievalViT(img_size=224, patch_size=16,
                 num_frames=1, tubelet_size=1,
                 in_chans=6, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4., norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                 norm_pix_loss=False, weights=Prithvi_100M_filepath)

    # 模型图像输入测试(B, D)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vit_model = vit_model.eval().to(device)
    img = torch.randn(1, 6, 1, 224, 224).to(device)
    print(vit_model)