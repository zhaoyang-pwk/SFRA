from net.modules import *
import torch
from net.encoder import SFRABlock, AdaptiveModulator, is_snreq_model, get_snreq_kwargs, snreq_mode_from_model, should_use_snreq, make_snreq_kwargs, should_use_sfra, make_sfra_kwargs


class BasicLayer(nn.Module):

    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, upsample=None, snr_eq_kwargs=None, sfra_kwargs=None,):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SFRABlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 norm_layer=norm_layer,
                                 snr_eq_kwargs=snr_eq_kwargs,
                                 sfra_kwargs=sfra_kwargs)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = upsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x, snr=None):
        for _, blk in enumerate(self.blocks):
            x = blk(x, snr=snr)

        if self.upsample is not None:
            x = self.upsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
            print("blk.flops()", blk.flops())
        if self.upsample is not None:
            flops += self.upsample.flops()
            print("upsample.flops()", self.upsample.flops())
        return flops

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        for _, blk in enumerate(self.blocks):
            blk.input_resolution = (H, W)
            blk.update_mask()
        if self.upsample is not None:
            self.upsample.input_resolution = (H, W)


class SwinJSCC_Decoder(nn.Module):
    def __init__(self, model, img_size, embed_dims, depths, num_heads, C,
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 bottleneck_dim=16, snr_eq_mode=None, snr_eq_scope='none',
                 snr_eq_hidden_dim=128, snr_scale=20.0, snr_emb_scale=0.1,
                 freq_enhance_mode='none', freq_enhance_scope='none', freq_reduction=4,
                 freq_snr_min=0.0, freq_snr_max=20.0, freq_residual_init=1.0,
                 freq_max_modulation=0.5):
        super().__init__()

        self.snr_eq_mode = snr_eq_mode if snr_eq_mode is not None else snreq_mode_from_model(model)
        self.snr_eq_scope = snr_eq_scope
        self.snr_eq_hidden_dim = int(snr_eq_hidden_dim)
        self.snr_scale = float(snr_scale)
        self.snr_emb_scale = float(snr_emb_scale)
        self.freq_enhance_mode = str(freq_enhance_mode or 'none')
        self.freq_enhance_scope = str(freq_enhance_scope or 'none')
        self.freq_reduction = int(freq_reduction)
        self.freq_snr_min = float(freq_snr_min)
        self.freq_snr_max = float(freq_snr_max)
        self.freq_residual_init = float(freq_residual_init)
        self.freq_max_modulation = float(freq_max_modulation)
        self.num_layers = len(depths)
        self.ape = ape
        self.embed_dims = embed_dims
        self.patch_norm = patch_norm
        self.num_features = bottleneck_dim
        self.mlp_ratio = mlp_ratio
        self.H = img_size[0]
        self.W = img_size[1]
        self.patches_resolution = (img_size[0] // 2 ** len(depths), img_size[1] // 2 ** len(depths))
        num_patches = self.H // 4 * self.W // 4
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[0]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dims[i_layer]),
                               out_dim=int(embed_dims[i_layer + 1]) if (i_layer < self.num_layers - 1) else 3,
                               input_resolution=(self.patches_resolution[0] * (2 ** i_layer),
                                                 self.patches_resolution[1] * (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               norm_layer=norm_layer,
                               upsample=PatchReverseMerging,
                               snr_eq_kwargs=make_snreq_kwargs(
                                   use_snreq=should_use_snreq(self.snr_eq_scope, i_layer, self.num_layers, 'decoder'),
                                   mode=self.snr_eq_mode,
                                   hidden_dim=self.snr_eq_hidden_dim,
                                   snr_scale=self.snr_scale,
                                   emb_scale=self.snr_emb_scale),
                               sfra_kwargs=make_sfra_kwargs(
                                   use_sfra=(self.freq_enhance_mode == 'sfra' and
                                             should_use_sfra(self.freq_enhance_scope, i_layer, self.num_layers, 'decoder')),
                                   reduction=self.freq_reduction,
                                   snr_min=self.freq_snr_min,
                                   snr_max=self.freq_snr_max,
                                   init_gamma=self.freq_residual_init,
                                   max_modulation=self.freq_max_modulation))
            self.layers.append(layer)
            print("Decoder ", layer.extra_repr())
        if C != None:
            self.head_list = nn.Linear(C, embed_dims[0])
        self.apply(self._init_weights)
        for module in self.modules():
            if hasattr(module, 'reset_sqa_identity'):
                module.reset_sqa_identity()
            if hasattr(module, 'reset_sfrc_identity'):
                module.reset_sfrc_identity()
        self.hidden_dim = int(self.embed_dims[0] * 1.5)
        self.layer_num = layer_num = 7
        if model != "SwinJSCC_w/_RA":
            self.bm_list = nn.ModuleList()
            self.sm_list = nn.ModuleList()
            self.sm_list.append(nn.Linear(self.embed_dims[0], self.hidden_dim))
            for i in range(layer_num):
                if i == layer_num - 1:
                    outdim = self.embed_dims[0]
                else:
                    outdim = self.hidden_dim
                self.bm_list.append(AdaptiveModulator(self.hidden_dim))
                self.sm_list.append(nn.Linear(self.hidden_dim, outdim))
            self.sigmoid = nn.Sigmoid()

    def forward(self, x, snr, model):
        if model == 'SwinJSCC_w/o_SAandRA':
            x = self.head_list(x)
            snr_for_attn = snr if ((self.snr_eq_mode != 'none' and self.snr_eq_scope != 'none') or (self.freq_enhance_mode == 'sfra' and self.freq_enhance_scope != 'none')) else None
            for i_layer, layer in enumerate(self.layers):
                x = layer(x, snr=snr_for_attn)
            B, L, N = x.shape
            x = x.reshape(B, self.H, self.W, N).permute(0, 3, 1, 2)
            return x

        elif is_snreq_model(model):
            x = self.head_list(x)
            snr_for_attn = snr if ((self.snr_eq_mode != 'none' and self.snr_eq_scope != 'none') or (self.freq_enhance_mode == 'sfra' and self.freq_enhance_scope != 'none')) else None
            for i_layer, layer in enumerate(self.layers):
                x = layer(x, snr=snr_for_attn)
            B, L, N = x.shape
            x = x.reshape(B, self.H, self.W, N).permute(0, 3, 1, 2)
            return x

        elif model == 'SwinJSCC_w/_SA':
            B, L, C = x.size()
            device = x.device
            x = self.head_list(x)
            snr_val = float(snr.detach().mean()) if torch.is_tensor(snr) else float(snr)
            snr_batch = torch.full((B, 1), snr_val, device=device)
            for i in range(self.layer_num):
                if i == 0:
                    temp = self.sm_list[i](x.detach())
                else:
                    temp = self.sm_list[i](temp)
                bm = self.bm_list[i](snr_batch).unsqueeze(1).expand(-1, L, -1)
                temp = temp * bm
            mod_val = self.sigmoid(self.sm_list[-1](temp))
            x = x * mod_val
            for i_layer, layer in enumerate(self.layers):
                x = layer(x)
            B, L, N = x.shape
            x = x.reshape(B, self.H, self.W, N).permute(0, 3, 1, 2)
            return x

        elif model == 'SwinJSCC_w/_RA':
            for i_layer, layer in enumerate(self.layers):
                x = layer(x)
            B, L, N = x.shape
            x = x.reshape(B, self.H, self.W, N).permute(0, 3, 1, 2)
            return x

        elif model == 'SwinJSCC_w/_SAandRA':
            B, L, C = x.size()
            device = x.device
            snr_val = float(snr.detach().mean()) if torch.is_tensor(snr) else float(snr)
            snr_batch = torch.full((B, 1), snr_val, device=device)
            for i in range(self.layer_num):
                if i == 0:
                    temp = self.sm_list[i](x.detach())
                else:
                    temp = self.sm_list[i](temp)
                bm = self.bm_list[i](snr_batch).unsqueeze(1).expand(-1, L, -1)
                temp = temp * bm
            mod_val = self.sigmoid(self.sm_list[-1](temp))
            x = x * mod_val
            for i_layer, layer in enumerate(self.layers):
                x = layer(x)
            B, L, N = x.shape
            x = x.reshape(B, self.H, self.W, N).permute(0, 3, 1, 2)
            return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def flops(self):
        flops = 0
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        return flops

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        self.H = H * 2 ** len(self.layers)
        self.W = W * 2 ** len(self.layers)
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H * (2 ** i_layer),
                                    W * (2 ** i_layer))


def create_decoder(**kwargs):
    model = SwinJSCC_Decoder(**kwargs)
    return model



def build_model(config):
    input_image = torch.ones([1, 1536, 256]).to(config.device)
    model = create_decoder(**config.encoder_kwargs).to(config.device)
    t0 = datetime.datetime.now()
    with torch.no_grad():
        for i in range(100):
            features = model(input_image, SNR=15)
        t1 = datetime.datetime.now()
        delta_t = t1 - t0
        print("Decoding Time per img {}s".format((delta_t.seconds + 1e-6 * delta_t.microseconds) / 100))
    print("TOTAL FLOPs {}G".format(model.flops() / 10 ** 9))
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("TOTAL Params {}M".format(num_params / 10 ** 6))

