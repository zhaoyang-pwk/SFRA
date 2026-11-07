from net.modules import *
import torch


def is_snreq_model(model):
    """Backward-compatible detector for legacy SNR_EQ_JSCC model names."""
    return str(model).startswith('SNR_EQ_JSCC')


def snreq_mode_from_model(model):
    model = str(model)
    if model == 'SNR_EQ_JSCC':
        return 'emb_caq'
    if model == 'SNR_EQ_JSCC_w/o_CAQ':
        return 'emb'
    if model == 'SNR_EQ_JSCC_w/o_EM':
        return 'caq'
    if model == 'SNR_EQ_JSCC_w/o_CA':
        return 'none'
    return 'none'


def should_use_snreq(scope, i_layer, num_layers, side):
    scope = str(scope or 'none')
    if scope == 'none':
        return False
    if scope == 'all':
        return True
    if side == 'encoder':
        if scope == 'bottleneck':
            return i_layer == num_layers - 1
        if scope == 'channel_side2':
            return i_layer >= num_layers - 2
    elif side == 'decoder':
        if scope == 'bottleneck':
            return i_layer == 0
        if scope == 'channel_side2':
            return i_layer <= 1
    return False



def should_use_sfra(scope, i_layer, num_layers, side):
    """Legacy-named selector for the SFRC branch; mirrors SQA scope."""
    scope = str(scope or 'none')
    if scope == 'none':
        return False
    if scope == 'all':
        return True
    if side == 'encoder':
        if scope == 'bottleneck':
            return i_layer == num_layers - 1
        if scope == 'channel_side2':
            return i_layer >= num_layers - 2
    elif side == 'decoder':
        if scope == 'bottleneck':
            return i_layer == 0
        if scope == 'channel_side2':
            return i_layer <= 1
    return False


def make_sfra_kwargs(use_sfra=False, reduction=4, snr_min=0.0, snr_max=20.0,
                     init_gamma=1.0, max_modulation=0.5):
    return dict(
        use_sfra=bool(use_sfra),
        sfra_reduction=int(reduction),
        sfra_snr_min=float(snr_min),
        sfra_snr_max=float(snr_max),
        sfra_init_gamma=float(init_gamma),
        sfra_max_modulation=float(max_modulation),
    )


def make_snreq_kwargs(use_snreq=False, mode='none', hidden_dim=128, snr_scale=20.0, emb_scale=0.1):
    mode = str(mode or 'none')
    use = bool(use_snreq and mode != 'none')
    return dict(
        use_snreq=use,
        use_caq=use and mode in ('caq', 'emb_caq'),
        use_snr_embedding=use and mode in ('emb', 'emb_caq'),
        snr_eq_hidden_dim=int(hidden_dim),
        snr_scale=float(snr_scale),
        snr_emb_scale=float(emb_scale),
    )


def get_snreq_kwargs(model):
    """Compatibility helper for old code paths. New code should pass args-derived kwargs."""
    mode = snreq_mode_from_model(model)
    return make_snreq_kwargs(use_snreq=(mode != 'none'), mode=mode)


class SFRABlock(nn.Module):
    """Swin block with optional SQA attention and an optional SFRC branch.

    Legacy argument and registered-attribute names are retained for compatibility.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,

                 mlp_ratio=4., qkv_bias=True, qk_scale=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, snr_eq_kwargs=None, sfra_kwargs=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = SQAWindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, **(snr_eq_kwargs or {}))

        self.norm2 = norm_layer(dim)
        self.norm_freq = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

        sfra_kwargs = sfra_kwargs or {}
        self.use_sfra = bool(sfra_kwargs.get('use_sfra', False))
        if self.use_sfra:
            self.freq_adapter = SFRC(
                dim=dim,
                input_resolution=self.input_resolution,
                reduction=sfra_kwargs.get('sfra_reduction', 4),
                snr_min=sfra_kwargs.get('sfra_snr_min', 0.0),
                snr_max=sfra_kwargs.get('sfra_snr_max', 20.0),
                init_gamma=sfra_kwargs.get('sfra_init_gamma', 1.0),
                max_modulation=sfra_kwargs.get('sfra_max_modulation', 0.5),
            )
        else:
            self.freq_adapter = None

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, snr=None):

        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        B_, N, C = x_windows.shape

        # merge windows
        attn_windows = self.attn(x_windows,
                                 add_token=False,
                                 mask=self.attn_mask,
                                 snr=snr)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN + internal frequency residual enhancement.
        # SFRC does FFT -> SNR-conditioned selection -> iFFT -> residual fusion internally;
        # it does not alter transmitted latent size or channel uses.
        x = shortcut + x
        if self.freq_adapter is not None:
            x = x + self.freq_adapter(self.norm_freq(x), snr=snr)
        x = x + self.mlp(self.norm2(x))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

    def update_mask(self):
        if self.freq_adapter is not None:
            self.freq_adapter.set_input_resolution(self.input_resolution)
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self.attn_mask = attn_mask.to(device=self.attn.relative_position_bias_table.device)
        else:
            pass


# Legacy import name; both names refer to exactly the same block class.
SwinTransformerBlock = SFRABlock


class BasicLayer(nn.Module):
    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm,
                 downsample=None, snr_eq_kwargs=None, sfra_kwargs=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = nn.ModuleList([
            SFRABlock(dim=out_dim,
                                 input_resolution=(input_resolution[0] // 2, input_resolution[1] // 2),
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 norm_layer=norm_layer,
                                 snr_eq_kwargs=snr_eq_kwargs,
                                 sfra_kwargs=sfra_kwargs)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, snr=None):
        if self.downsample is not None:
            x = self.downsample(x)
        for _, blk in enumerate(self.blocks):
            x = blk(x, snr=snr)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def update_resolution(self, H, W):
        for _, blk in enumerate(self.blocks):
            blk.input_resolution = (H, W)
            blk.update_mask()
        if self.downsample is not None:
            self.downsample.input_resolution = (H * 2, W * 2)

class AdaptiveModulator(nn.Module):
    def __init__(self, M):
        super(AdaptiveModulator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.Sigmoid()
        )

    def forward(self, snr):
        return self.fc(snr)

class SwinJSCC_Encoder(nn.Module):
    def __init__(self, model, img_size, patch_size, in_chans,
                 embed_dims, depths, num_heads, C,
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, patch_norm=True,
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
        self.patch_norm = patch_norm
        self.num_features = bottleneck_dim
        self.mlp_ratio = mlp_ratio
        self.embed_dims = embed_dims
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.patches_resolution = img_size
        self.H = img_size[0] // (2 ** self.num_layers)
        self.W = img_size[1] // (2 ** self.num_layers)
        self.patch_embed = PatchEmbed(img_size, 2, 3, embed_dims[0])
        self.hidden_dim = int(self.embed_dims[len(embed_dims)-1] * 1.5)
        self.layer_num = layer_num = 7

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dims[i_layer - 1]) if i_layer != 0 else 3,
                               out_dim=int(embed_dims[i_layer]),
                               input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                                 self.patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               norm_layer=norm_layer,
                               downsample=PatchMerging if i_layer != 0 else None,
                               snr_eq_kwargs=make_snreq_kwargs(
                                   use_snreq=should_use_snreq(self.snr_eq_scope, i_layer, self.num_layers, 'encoder'),
                                   mode=self.snr_eq_mode,
                                   hidden_dim=self.snr_eq_hidden_dim,
                                   snr_scale=self.snr_scale,
                                   emb_scale=self.snr_emb_scale),
                               sfra_kwargs=make_sfra_kwargs(
                                   use_sfra=(self.freq_enhance_mode == 'sfra' and
                                             should_use_sfra(self.freq_enhance_scope, i_layer, self.num_layers, 'encoder')),
                                   reduction=self.freq_reduction,
                                   snr_min=self.freq_snr_min,
                                   snr_max=self.freq_snr_max,
                                   init_gamma=self.freq_residual_init,
                                   max_modulation=self.freq_max_modulation))
            print("Encoder ", layer.extra_repr())
            self.layers.append(layer)
        self.norm = norm_layer(embed_dims[-1])
        if C != None:
            self.head_list = nn.Linear(embed_dims[-1], C)
        self.apply(self._init_weights)
        for module in self.modules():
            if hasattr(module, 'reset_sqa_identity'):
                module.reset_sqa_identity()
            if hasattr(module, 'reset_sfrc_identity'):
                module.reset_sfrc_identity()
        # Channel ModNet and Rate  ModNet
        if model == 'SwinJSCC_w/_SAandRA':
            self.bm_list = nn.ModuleList()
            self.sm_list = nn.ModuleList()
            self.sm_list.append(nn.Linear(self.embed_dims[len(embed_dims) - 1], self.hidden_dim))
            for i in range(layer_num):
                if i == layer_num - 1:
                    outdim = self.embed_dims[len(embed_dims) - 1]
                else:
                    outdim = self.hidden_dim
                self.bm_list.append(AdaptiveModulator(self.hidden_dim))
                self.sm_list.append(nn.Linear(self.hidden_dim, outdim))
            self.sigmoid = nn.Sigmoid()

            self.bm_list1 = nn.ModuleList()
            self.sm_list1 = nn.ModuleList()
            self.sm_list1.append(nn.Linear(self.embed_dims[len(embed_dims) - 1], self.hidden_dim))
            for i in range(layer_num):
                if i == layer_num - 1:
                    outdim = self.embed_dims[len(embed_dims) - 1]
                else:
                    outdim = self.hidden_dim
                self.bm_list1.append(AdaptiveModulator(self.hidden_dim))
                self.sm_list1.append(nn.Linear(self.hidden_dim, outdim))
            self.sigmoid1 = nn.Sigmoid()
        else:
            self.bm_list = nn.ModuleList()
            self.sm_list = nn.ModuleList()
            self.sm_list.append(nn.Linear(self.embed_dims[len(embed_dims) - 1], self.hidden_dim))
            for i in range(layer_num):
                if i == layer_num - 1:
                    outdim = self.embed_dims[len(embed_dims) - 1]
                else:
                    outdim = self.hidden_dim
                self.bm_list.append(AdaptiveModulator(self.hidden_dim))
                self.sm_list.append(nn.Linear(self.hidden_dim, outdim))
            self.sigmoid = nn.Sigmoid()



    def forward(self, x, snr, rate, model):
        B, C, H, W = x.size()
        device = x.device
        x = self.patch_embed(x)
        snr_for_attn = snr if ((self.snr_eq_mode != 'none' and self.snr_eq_scope != 'none') or (self.freq_enhance_mode == 'sfra' and self.freq_enhance_scope != 'none')) else None
        for i_layer, layer in enumerate(self.layers):
            x = layer(x, snr=snr_for_attn)
        x = self.norm(x)

        if is_snreq_model(model):
            x = self.head_list(x)
            return x

        if model == 'SwinJSCC_w/o_SAandRA':
            x = self.head_list(x)
            return x

        elif model == 'SwinJSCC_w/_SA':
            snr_val = float(snr.detach().mean()) if torch.is_tensor(snr) else float(snr)
            snr_batch = torch.full((B, 1), snr_val, device=device)
            for i in range(self.layer_num):
                if i == 0:
                    temp = self.sm_list[i](x.detach())
                else:
                    temp = self.sm_list[i](temp)

                bm = self.bm_list[i](snr_batch).unsqueeze(1).expand(-1, x.size(1), -1)
                temp = temp * bm
            mod_val = self.sigmoid(self.sm_list[-1](temp))
            x = x * mod_val
            x = self.head_list(x)
            return x

        elif model == 'SwinJSCC_w/_RA':
            rate_val = float(rate.detach().mean()) if torch.is_tensor(rate) else float(rate)
            rate_batch = torch.full((B, 1), rate_val, device=device)
            for i in range(self.layer_num):
                if i == 0:
                    temp = self.sm_list[i](x.detach())
                else:
                    temp = self.sm_list[i](temp)

                bm = self.bm_list[i](rate_batch).unsqueeze(1).expand(-1, x.size(1), -1)
                temp = temp * bm
            mod_val = self.sigmoid(self.sm_list[-1](temp))
            x = x * mod_val
            mask = torch.sum(mod_val, dim=1)
            sorted, indices = mask.sort(dim=1, descending=True)
            c_indices = indices[:, :rate]
            add = torch.arange(0, B * x.size(2), x.size(2), device=device).unsqueeze(1).repeat(1, rate)
            c_indices = c_indices + add.long()
            mask = torch.zeros(mask.size(), device=device).reshape(-1)
            mask[c_indices.reshape(-1)] = 1
            mask = mask.reshape(B, x.size()[2])
            mask = mask.unsqueeze(1).expand(-1, x.size(1), -1)
            x = x * mask
            return x, mask

        elif model == 'SwinJSCC_w/_SAandRA':
            snr_cuda = torch.tensor(snr, dtype=torch.float, device=device)
            rate_cuda = torch.tensor(rate, dtype=torch.float, device=device)
            snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
            rate_batch = rate_cuda.unsqueeze(0).expand(B, -1)
            for i in range(self.layer_num):
                if i == 0:
                    temp = self.sm_list1[i](x.detach())
                else:
                    temp = self.sm_list1[i](temp)

                bm = self.bm_list1[i](snr_batch).unsqueeze(1).expand(-1, x.size(1), -1)
                temp = temp * bm
            mod_val1 = self.sigmoid1(self.sm_list1[-1](temp))
            x = x * mod_val1

            for i in range(self.layer_num):
                if i == 0:
                    temp = self.sm_list[i](x.detach())
                else:
                    temp = self.sm_list[i](temp)

                bm = self.bm_list[i](rate_batch).unsqueeze(1).expand(-1, x.size(1), -1)
                temp = temp * bm
            mod_val = self.sigmoid(self.sm_list[-1](temp))
            x = x * mod_val
            mask = torch.sum(mod_val, dim=1)
            sorted, indices = mask.sort(dim=1, descending=True)
            c_indices = indices[:, :rate]
            add = torch.arange(0, B * x.size(2), x.size(2), device=device).unsqueeze(1).repeat(1, rate)
            c_indices = c_indices + add.long()
            mask = torch.zeros(mask.size(), device=device).reshape(-1)
            mask[c_indices.reshape(-1)] = 1
            mask = mask.reshape(B, x.size()[2])
            mask = mask.unsqueeze(1).expand(-1, x.size(1), -1)

            x = x * mask
            return x, mask

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
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        return flops

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H // (2 ** (i_layer + 1)),
                                    W // (2 ** (i_layer + 1)))


def create_encoder(**kwargs):
    model = SwinJSCC_Encoder(**kwargs)
    return model


def build_model(config):
    input_image = torch.ones([1, 256, 256]).to(config.device)
    model = create_encoder(**config.encoder_kwargs)
    model(input_image)
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("TOTAL Params {}M".format(num_params / 10 ** 6))
    print("TOTAL FLOPs {}G".format(model.flops() / 10 ** 9))
