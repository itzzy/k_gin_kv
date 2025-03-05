# Acknowledgement
# This part of code is developed based on the repository MAE: https://github.com/facebookresearch/mae.

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block, PatchEmbed,Mlp
from utils.model_related import get_2d_sincos_pos_embed
from utils import ifft2c

'''
本代码会报以下错误，还未解决--1118
forward-img type: <class 'torch.Tensor'>, shape: torch.Size([32, 18, 192, 192])
forward-mask type: <class 'torch.Tensor'>, shape: torch.Size([32, 18, 192])
forward-img_0F-1 type: <class 'torch.Tensor'>, shape: torch.Size([32, 32, 192, 192])
forward-img_0F-2 type: <class 'torch.Tensor'>, shape: torch.Size([32, 384, 32, 192])
  0%|                                                                                                                                                                    | 0/2 [00:11<?, ?it/s]
Traceback (most recent call last):
  File "train.py", line 66, in <module>
    trainer.run()
  File "/data0/zhiyong/code/github/itzzy_git/k-gin_kv/trainer.py", line 101, in run
    self.train_one_epoch(epoch)
  File "/data0/zhiyong/code/github/itzzy_git/k-gin_kv/trainer.py", line 122, in train_one_epoch
    k_recon_2ch, im_recon = self.network(kspace, mask=sampling_mask)  # size of kspace and mask: [B, T, H, W]
  File "/home/zhiyongzhang/anaconda3/envs/k_gin/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/zhiyongzhang/anaconda3/envs/k_gin/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data0/zhiyong/code/github/itzzy_git/k-gin_kv/model/k_interpolator.py", line 348, in forward
    kv, _ = self.encoder(img_0F, mask_0F)
  File "/data0/zhiyong/code/github/itzzy_git/k-gin_kv/model/k_interpolator.py", line 201, in encoder
    kspace = kspace + self.pos_embed[:, 1:, :]
RuntimeError: The size of tensor a (6144) must match the size of tensor b (192) at non-singleton dimension 1
'''

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        #self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, kv, x):
        B, N, C = x.shape
        #print('kv, x', kv.shape, x.shape) kv, x torch.Size([1, 3457, 512]) torch.Size([1, 3457, 512])
        #qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        #q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        #qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k = kv.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        v = k
        q = x.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # x: (B, N, C)
        return x
        
class DecoderTrans(nn.Module):
    #def __init__(self, dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, act_layer=act_layer):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, kv, x):
        #x = x + self.drop_path(self.attn(self.norm1(x)))
        #x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x + self.drop_path(self.attn(kv,x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))        
        return x


class KInterpolator(nn.Module):
    def __init__(self, config):
        super().__init__()
        config = config.KInterpolator
        self.img_size = config.img_size
        self.in_chans = config.in_chans
        self.embed_dim = config.embed_dim
        depth = config.depth
        num_heads = config.num_heads
        self.decoder_embed_dim = config.decoder_embed_dim
        decoder_depth = config.decoder_depth
        decoder_num_heads = config.decoder_num_heads
        mlp_ratio = config.mlp_ratio
        norm_layer = eval(config.norm_layer)
        act_layer = eval(config.act_layer)
        self.xt_y_tuning = config.xt_y_tuning
        self.yt_x_tuning = config.yt_x_tuning
        self.ref_repl_prior_denoiser = config.ref_repl_prior_denoiser
        self.post_tuning = True if self.xt_y_tuning or self.yt_x_tuning else False
        self.xy_t_patch_tuning = config.xy_t_patch_tuning

        self.num_patches = self.img_size[0] * self.img_size[1]

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, self.img_size[1] + 1, self.embed_dim), requires_grad=False)  # fixed sin-cos embedding

        B = torch.randn((1, 1, self.embed_dim//2), dtype=torch.float32)
        self.register_buffer('B', B)

        #self.patch_embed = nn.Conv2d(self.img_size[2]*2, self.embed_dim, kernel_size=(1, 1))
        #self.patch_embed = nn.Conv2d(self.img_size[2]*2, self.embed_dim, 1, 1)
        self.patch_embed = nn.Linear(self.img_size[2]*2, self.embed_dim, bias=True)

        self.blocks = nn.ModuleList([
            Block(self.embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])            
        self.norm = norm_layer(self.embed_dim)

        self.decoder_embed = nn.Linear(self.embed_dim, self.decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.img_size[1] + 1, self.decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            DecoderTrans(self.decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, act_layer=act_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(self.decoder_embed_dim)

        self.decoder_pred = nn.Linear(self.decoder_embed_dim, self.img_size[2]*2, bias=True)

        if self.yt_x_tuning:
            self.yt_x_num_patches = self.num_patches
            self.yt_x_pos_embed = nn.Parameter(torch.zeros(1, self.yt_x_num_patches, config.yt_x_embed_dim),
                                               requires_grad=False)
            self.yt_x_patch_embed = nn.Conv2d(self.img_size[2] * 2, config.yt_x_embed_dim, kernel_size=(1, 1))
            self.yt_x_blocks = nn.ModuleList([
                Block(config.yt_x_embed_dim, config.yt_x_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                      act_layer=act_layer) for i in range(config.yt_x_depth)])
            self.yt_x_norm = norm_layer(config.yt_x_embed_dim)
            self.yt_x_pred = nn.Linear(config.yt_x_embed_dim, self.img_size[2] * 2,
                                       bias=True)
        if self.xt_y_tuning:
            self.xt_y_num_patches = self.img_size[0] * self.img_size[2]
            self.xt_y_pos_embed = nn.Parameter(torch.zeros(1, self.xt_y_num_patches, config.xt_y_embed_dim), requires_grad=False)  # fixed sin-cos embedding
            self.xt_y_patch_embed = nn.Conv2d(self.img_size[1]*2, config.xt_y_embed_dim, kernel_size=(1, 1))

            self.xt_y_blocks = nn.ModuleList([
                Block(config.xt_y_embed_dim, config.xt_y_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, act_layer=act_layer)
                for i in range(config.xt_y_depth)])
            self.xt_y_norm = norm_layer(config.xt_y_embed_dim)
            self.xt_y_pred = nn.Linear(config.xt_y_embed_dim, self.img_size[1]*2, bias=True)
        if self.xy_t_patch_tuning:
            self.xy_t_patch_embed = PatchEmbed(self.img_size[-1:0:-1], config.patch_size, self.img_size[0]*2, config.xy_t_patch_embed_dim)
            self.xy_t_patch_pos_embed = nn.Parameter(torch.zeros(1, self.xy_t_patch_embed.num_patches, config.xy_t_patch_embed_dim), requires_grad=False)
            self.xy_t_patch_blocks = nn.ModuleList([
                Block(config.xy_t_patch_embed_dim, config.xy_t_patch_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, act_layer=act_layer)
                for i in range(config.xy_t_patch_depth)])
            self.xy_t_patch_norm = norm_layer(config.xy_t_patch_embed_dim)
            self.xy_t_patch_pred = nn.Linear(config.xy_t_patch_embed_dim, config.patch_size**2*self.img_size[0]*2, bias=True)

        self.fc_norm = norm_layer(32)

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], 1, self.img_size[1], cls_token=True) #self.img_size[0]
        self.pos_embed.data.copy_(torch.tensor(pos_embed).unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], 1, self.img_size[1], cls_token=True) #self.img_size[0]
        self.decoder_pos_embed.data.copy_(torch.tensor(decoder_pos_embed).unsqueeze(0))

        if self.xt_y_tuning:
            xt_y_pos_embed = get_2d_sincos_pos_embed(self.xt_y_pos_embed.shape[-1], self.img_size[0], self.img_size[2], cls_token=False)
            self.xt_y_pos_embed.data.copy_(torch.tensor(xt_y_pos_embed).unsqueeze(0))

        if self.yt_x_tuning:
            yt_x_pos_embed = get_2d_sincos_pos_embed(self.yt_x_pos_embed.shape[-1], self.img_size[0], self.img_size[1], cls_token=False)
            self.yt_x_pos_embed.data.copy_(torch.tensor(yt_x_pos_embed).unsqueeze(0))

        if self.xy_t_patch_tuning:
            xy_t_patch_pos_embed = get_2d_sincos_pos_embed(self.xy_t_patch_pos_embed.shape[-1], self.xy_t_patch_embed.grid_size[0], self.xy_t_patch_embed.grid_size[1], cls_token=False)
            self.xy_t_patch_pos_embed.data.copy_(torch.tensor(xy_t_patch_pos_embed).unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def encoder(self, kspace, mask):
        b, c, h, w = kspace.shape
        kspace = kspace.flatten(2).transpose(1, 2)
        kspace = self.patch_embed(kspace)

        #kspace = kspace.flatten(2).transpose(1, 2)  # BCHW -> BNC
        #print('k, m', kspace.shape, mask.shape)
        kspace = kspace + self.pos_embed[:, 1:, :]

        kspace = kspace[mask > 0, :].reshape(b, -1, self.embed_dim)
        ids_shuffle = torch.argsort(mask, dim=1, descending=True)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(kspace.shape[0], -1, -1)
        kspace = torch.cat((cls_tokens, kspace), dim=1)
        
#        imgadd = torch.mean(tensor, dim=2, keepdim=True)
#        imgadd = torch.cat((mean_of_third_dim, tensor), dim=2)

        # apply Transformer blocks
        for blk in self.blocks:
            kspace = blk(kspace)
        kspace = self.norm(kspace)

        return kspace, ids_restore

    def unpatchify_xy_t(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.xy_t_patch_embed.patch_size[0]
        h, w = self.xy_t_patch_embed.grid_size[0], self.xy_t_patch_embed.grid_size[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.img_size[0]*2))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.img_size[0], 2, h * p, w * p))
        imgs = torch.einsum('btchw->bthwc', imgs)
        return imgs

    def decoder(self, kv, q, ids_restore, mask):
        #print('kv, q', kv.shape, q.shape) #kv, q torch.Size([1, 3457, 512]) torch.Size([1, 919, 512]) 
        kspace = self.decoder_embed(q)

        mask_tokens = self.mask_token.repeat(kspace.shape[0], ids_restore.shape[1] + 1 - kspace.shape[1], 1)
        kspace_full = torch.cat([kspace[:, 1:, :], mask_tokens], dim=1)  # no cls token
        kspace_full = torch.gather(kspace_full, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, kspace.shape[2]))  # unshuffle
        kspace = torch.cat([kspace[:, :1, :], kspace_full], dim=1)  # append cls token

        # add pos embed
        kspace = kspace + self.decoder_pos_embed
        #print('kspace', kspace.shape) kspace torch.Size([1, 3457, 512])
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            kspace = blk(kv, kspace)
        kspace = self.decoder_norm(kspace)

        latent_decoder = kspace[:, 1:, :][mask==0, :].reshape(kspace.shape[0], -1, self.decoder_embed_dim)

        # predictor projection
        kspace = self.decoder_pred(kspace)

        # remove cls token
        kspace = kspace[:, 1:, :]

        return kspace, latent_decoder

    def xt_y(self, kspace):
        b, t, h, w, c = kspace.shape
        kspace = torch.einsum('bthwc->bwcth', kspace).flatten(1,2)
        kspace = self.xt_y_patch_embed(kspace)
        kspace = kspace.flatten(2).transpose(1, 2)  # BCN -> BNC
        kspace = kspace + self.xt_y_pos_embed
        for blk in self.xt_y_blocks:
            kspace = blk(kspace)
        kspace = self.xt_y_norm(kspace)
        kspace = self.xt_y_pred(kspace)
        return kspace.reshape((b, t, h, w, 2))

    def x_yt(self, kspace):
        b, t, h, w, c = kspace.shape
        kspace = torch.einsum('bthwc->btwch', kspace).flatten(1,3)
        kspace = self.x_yt_patch_embed(kspace)
        kspace = kspace.transpose(1, 2)  # BCN -> BNC
        kspace = kspace + self.x_yt_pos_embed
        for blk in self.x_yt_blocks:
            kspace = blk(kspace)
        kspace = self.x_yt_norm(kspace)
        kspace = self.x_yt_pred(kspace)

        kspace = kspace.reshape((b, h, t, w, 2))
        return torch.einsum('bhtwc->bthwc', kspace)

    def yt_x(self, kspace):
        b, t, h, w, c = kspace.shape
        kspace = torch.einsum('bthwc->bhctw', kspace).flatten(1, 2)
        kspace = self.yt_x_patch_embed(kspace)
        kspace = kspace.flatten(2).transpose(1, 2)
        kspace = kspace + self.yt_x_pos_embed
        for blk in self.yt_x_blocks:
            kspace = blk(kspace)
        kspace = self.yt_x_norm(kspace)
        kspace = self.yt_x_pred(kspace)
        kspace = kspace.reshape((b, t, w, h, 2))

        return torch.einsum('btwhc->bthwc', kspace)

    def xy_t_patch(self, kspace):
        b, t, h, w, c = kspace.shape
        kspace = torch.einsum('bthwc->btchw', kspace).flatten(1, 2)
        kspace = self.xy_t_patch_embed(kspace)


        kspace = kspace + self.xy_t_patch_pos_embed
        for blk in self.xy_t_patch_blocks:
            kspace = blk(kspace)
        kspace = self.xy_t_patch_norm(kspace)
        kspace = self.xy_t_patch_pred(kspace)
        kspace = self.unpatchify_xy_t(kspace)
        return kspace.contiguous()

    def forward(self, img, mask):
        B, T, H, W = img.shape
        # size of input img and mask: [B, T, H, W]
        #print('k, m', img.shape, mask.shape) k, m torch.Size([1, 18, 192, 192]) torch.Size([1, 18, 192])
        # forward-img type: <class 'torch.Tensor'>, shape: torch.Size([32, 18, 192, 192])
        # forward-mask type: <class 'torch.Tensor'>, shape: torch.Size([32, 18, 192])
        # forward-img_0F-1 type: <class 'torch.Tensor'>, shape: torch.Size([32, 32, 192, 192])
        # forward-img_0F-2 type: <class 'torch.Tensor'>, shape: torch.Size([32, 384, 32, 192])
        print(f"forward-img type: {type(img)}, shape: {img.shape}")
        print(f"forward-mask type: {type(mask)}, shape: {mask.shape}") 
        img_0F = img.sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)
        print(f"forward-img_0F-1 type: {type(img_0F)}, shape: {img_0F.shape}")
        #img_0F = img_0F.repeat(1, 1, 1, 1)
        #img_0F = img_0F.repeat(1, img.shape[1], 1, 1)
        #torch.mean(img, dim=1, keepdim=True).repeat(1, img.shape[1], 1, 1)
                
        #mask_0F = torch.ones(mask.shape[0], mask.shape[1], mask.shape[2])
        mask_0F = torch.ones(mask.shape[0], 1, mask.shape[2])
        
        
        
        img_orig = torch.view_as_real(img)       
        mask_orig = mask[..., None, None].expand_as(img_orig) #torch.Size([1, 18, 192, 192, 2])
        
        img = torch.view_as_real(torch.einsum('bthw->btwh', img)).flatten(-2)
        img = torch.einsum('bhwt->bthw', img)
        img_0F = torch.view_as_real(torch.einsum('bthw->btwh', img_0F)).flatten(-2)
        img_0F = torch.einsum('bhwt->bthw', img_0F)
        print(f"forward-img_0F-2 type: {type(img_0F)}, shape: {img_0F.shape}")
        b, h_2, t, w = img.shape
        predt_list = []
        for time in range(T):
            maskt = mask[:, time, :].unsqueeze(1).flatten(1, -1)
            mask_0F = mask_0F.flatten(1, -1) 
            imgt = img[:, :, time, :].unsqueeze(2)
            #print('k, m', img_0F.shape, mask_0F.shape) #k, m torch.Size([1, 384, 1, 192]) torch.Size([1, 192])
            kv, _ = self.encoder(img_0F, mask_0F)
            #print('k, m', imgt.shape, maskt.shape) #k, m torch.Size([1, 384, 1, 192]) torch.Size([1, 192])
            q, ids_restore = self.encoder(imgt, maskt)
            predt, latent_decoder = self.decoder(kv, q, ids_restore, maskt)
            #print('predt', predt.shape) #predt torch.Size([1, 192, 384])
            predt_list.append(predt.unsqueeze(1))
            
        pred = torch.cat(predt_list, dim=1)
        #print('pred', pred.shape)  #pred pred torch.Size([1, 18, 192, 384])
        pred = pred.reshape((b, t, w, int(h_2/2), 2))
        pred = torch.einsum('btwhc->bthwc', pred)
        pred_list = [pred]

        pred_t = pred.clone()
        if self.ref_repl_prior_denoiser: pred_t[torch.where(mask_orig==1)] = img_orig[torch.where(mask_orig==1)]

        if self.yt_x_tuning:
            pred_t = self.yt_x(pred_t) + pred_t
            pred_list.append(pred_t)
        pred_t1 = pred_t.clone()
        if self.ref_repl_prior_denoiser: pred_t1[torch.where(mask_orig==1)] = img_orig[torch.where(mask_orig==1)]

        if self.xt_y_tuning:
            pred_t1 = self.xt_y(pred_t1) + pred_t1
            pred_list.append(pred_t1)
        pred_t2 = pred_t1.clone()
        if self.ref_repl_prior_denoiser: pred_t2[torch.where(mask_orig==1)] = img_orig[torch.where(mask_orig==1)]

        if self.xy_t_patch_tuning:
            pred_t2 = self.xy_t_patch(pred_t2) + pred_t2
            pred_list.append(pred_t2)
        pred_t3 = pred_t2.clone()
        if self.ref_repl_prior_denoiser: pred_t3[torch.where(mask_orig==1)] = img_orig[torch.where(mask_orig==1)]

        k_recon_complex = torch.view_as_complex(pred_t3)
        im_recon = ifft2c(k_recon_complex.to(torch.complex64))

        return pred_list, im_recon
