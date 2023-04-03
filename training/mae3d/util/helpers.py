import numpy as np
from torch import nn

# patch embedding
# ----------------------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    """ 3D Image to Patch Embedding
    """
    def __init__(self, img_size: tuple, patch_size: tuple, in_chans=1, embed_dim=768):
        super().__init__()
        assert len(img_size) == 3 and len(patch_size) == 3, f'''
            dimension != 3: img_size={img_size} patch_size={patch_size}'''

        self.img_size = np.array(img_size)
        self.patch_size = np.array(patch_size)

        assert not (self.img_size % self.patch_size).any(), f'''
            image size module patch size error'''

        self.dim_num_patches = (self.img_size // self.patch_size).astype(int)
        self.num_patches = np.prod(self.dim_num_patches)

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x): 
        B, C, H, W, D = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
# ----------------------------------------------------------------------------------------

# positional embedding    
# ----------------------------------------------------------------------------------------
def get_3d_sincos_pos_embed(embed_dim, grid_size: tuple, cls_token=False):
    """ 
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    assert len(grid_size) == 3, f'''
        dimension grid_size != 3: {grid_size}'''
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid_d = np.arange(grid_size[2], dtype=np.float32)
    # does d really go first here?
    grid = np.meshgrid(grid_d, grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size[0], grid_size[1], grid_size[2]])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 3 == 0, f'embed_dim = {embed_dim}'

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (H*W*D, d/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (H*W*D, d/3)
    emb_d = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (H*W*D, d/3)
    emb = np.concatenate([emb_h, emb_w, emb_d], axis=1) # (H*W*D, d)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
# ----------------------------------------------------------------------------------------
