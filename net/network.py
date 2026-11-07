from net.decoder import *
from net.encoder import *
from loss.distortion import Distortion
from net.channel import Channel
from random import choice
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mean_float(value):
    if torch.is_tensor(value):
        return float(value.detach().float().mean().item())
    return float(value)


class SwinJSCC(nn.Module):
    def __init__(self, args, config):
        super(SwinJSCC, self).__init__()
        self.args = args
        self.config = config
        encoder_kwargs = config.encoder_kwargs
        decoder_kwargs = config.decoder_kwargs
        self.encoder = create_encoder(**encoder_kwargs)
        self.decoder = create_decoder(**decoder_kwargs)
        if config.logger is not None:
            config.logger.info("Network config: ")
            config.logger.info("Encoder: ")
            config.logger.info(encoder_kwargs)
            config.logger.info("Decoder: ")
            config.logger.info(decoder_kwargs)
        self.distortion_loss = Distortion(args)
        self.channel = Channel(args, config)
        self.pass_channel = config.pass_channel
        self.squared_difference = torch.nn.MSELoss(reduction='none')
        self.H = self.W = 0
        self.multiple_snr = [float(i) for i in args.multiple_snr.split(",") if i != ""]
        self.channel_number = [int(i) for i in args.C.split(",") if i != ""]
        self.downsample = config.downsample
        self.model = args.model
        self.snr_eq_mode = getattr(args, 'snr_eq_mode', 'none')
        if is_snreq_model(self.model) and self.snr_eq_mode == 'none':
            self.snr_eq_mode = snreq_mode_from_model(self.model)
        self.snr_eq_scope = getattr(args, 'snr_eq_scope', 'none')
        self.snreq_lambda = float(getattr(args, 'snr_eq_penalty_lambda', getattr(args, 'snreq_lambda', 1e-5)))
        self.snr_sample_mode = getattr(args, 'snr_sample_mode', getattr(args, 'train_snr_mode', 'batch_choice'))
        self.train_snr_min = getattr(args, 'train_snr_min', None)
        self.train_snr_max = getattr(args, 'train_snr_max', None)
        self.fading_blocks = int(getattr(args, 'fading_blocks', getattr(args, 'rayleigh_fading_blocks', 8)))
        self.fading_min_h = float(getattr(args, 'fading_min_h', 1e-4))
        self.snr_feedback_mode = getattr(args, 'snr_feedback_mode', 'avg')
        self.last_snreq_penalty = None

    def distortion_loss_wrapper(self, x_gen, x_real):
        distortion_loss = self.distortion_loss.forward(x_gen, x_real, normalization=self.config.norm)
        return distortion_loss

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature

    def _update_resolution_if_needed(self, input_image):
        _, _, H, W = input_image.shape
        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
            self.H = H
            self.W = W

    def _sample_from_grid(self, B, device, dtype):
        grid = torch.tensor(self.multiple_snr, device=device, dtype=dtype)
        if grid.numel() == 0:
            grid = torch.tensor([10.0], device=device, dtype=dtype)
        idx = torch.randint(0, grid.numel(), (B,), device=device)
        return grid[idx]

    def _sample_uniform(self, B, device, dtype):
        if self.train_snr_min is None or self.train_snr_max is None:
            lo = min(self.multiple_snr)
            hi = max(self.multiple_snr)
        else:
            lo = float(self.train_snr_min)
            hi = float(self.train_snr_max)
        return torch.empty(B, device=device, dtype=dtype).uniform_(float(lo), float(hi))

    def _rayleigh_instant_snr(self, avg_snr_db, blocks):
        """Convert average SNR mu to M instantaneous SNRs using |h|^2.

        This gives the SNR-EQ/Jakes-style training signal without forcing the
        model into a separate physical Rayleigh channel branch. The channel layer
        subsequently applies blockwise AWGN according to these instantaneous SNRs.
        """
        avg = avg_snr_db.view(-1, 1)
        h2 = (torch.randn(avg.size(0), blocks, device=avg.device, dtype=avg.dtype) ** 2
              + torch.randn(avg.size(0), blocks, device=avg.device, dtype=avg.dtype) ** 2) / 2.0
        h2 = torch.clamp(h2, min=self.fading_min_h)
        gamma = torch.pow(10.0, avg / 10.0) * h2
        return 10.0 * torch.log10(torch.clamp(gamma, min=1e-12))

    def _make_snr_state(self, input_image, given_SNR=None):
        B = input_image.size(0)
        device = input_image.device
        dtype = input_image.dtype
        mode = self.snr_sample_mode

        if given_SNR is not None:
            if torch.is_tensor(given_SNR):
                avg_snr = given_SNR.to(device=device, dtype=dtype)
                if avg_snr.dim() == 0:
                    avg_snr = avg_snr.view(1).expand(B)
                elif avg_snr.numel() == 1:
                    avg_snr = avg_snr.flatten().expand(B)
                else:
                    avg_snr = avg_snr.flatten()[:B]
            else:
                avg_snr = torch.full((B,), float(given_SNR), device=device, dtype=dtype)
            if mode in ('fading_multi', 'rayleigh_snr_eq') and self.fading_blocks > 1:
                inst_snr = self._rayleigh_instant_snr(avg_snr, self.fading_blocks)
                return avg_snr, inst_snr
            return avg_snr, avg_snr

        if mode in ('batch_choice', 'random'):
            avg_snr = torch.full((B,), float(choice(self.multiple_snr)), device=device, dtype=dtype)
            return avg_snr, avg_snr
        if mode == 'per_image_choice':
            avg_snr = self._sample_from_grid(B, device, dtype)
            return avg_snr, avg_snr
        if mode == 'per_image_uniform':
            avg_snr = self._sample_uniform(B, device, dtype)
            return avg_snr, avg_snr
        if mode in ('fading_multi', 'rayleigh_snr_eq'):
            # Average SNR is still sampled per image; channel SNR is [B,M].
            if getattr(self.args, 'fading_avg_snr_mode', 'choice') == 'uniform':
                avg_snr = self._sample_uniform(B, device, dtype)
            else:
                avg_snr = self._sample_from_grid(B, device, dtype)
            inst_snr = self._rayleigh_instant_snr(avg_snr, self.fading_blocks)
            return avg_snr, inst_snr

        # Safe default.
        avg_snr = torch.full((B,), float(choice(self.multiple_snr)), device=device, dtype=dtype)
        return avg_snr, avg_snr

    def _snr_for_decoder(self, avg_snr, channel_snr):
        # Current SwinJSCC attention blocks are global over the feature map; using
        # average SNR as the conditioning signal is the stable default. The channel
        # still receives instantaneous block SNRs when channel_snr is [B,M].
        if self.snr_feedback_mode == 'instant_mean' and torch.is_tensor(channel_snr) and channel_snr.dim() == 2:
            return channel_snr.mean(dim=1)
        return avg_snr

    def forward(self, input_image, given_SNR=None, given_rate=None):
        self._update_resolution_if_needed(input_image)
        avg_snr, channel_snr = self._make_snr_state(input_image, given_SNR=given_SNR)

        if given_rate is None:
            channel_number = choice(self.channel_number)
        else:
            channel_number = given_rate

        encoder_snr = avg_snr
        decoder_snr = self._snr_for_decoder(avg_snr, channel_snr)

        if self.model == 'SwinJSCC_w/o_SAandRA' or self.model == 'SwinJSCC_w/_SA' or is_snreq_model(self.model):
            feature = self.encoder(input_image, encoder_snr, channel_number, self.model)
            CBR = feature.numel() / 2 / input_image.numel()
            if self.pass_channel:
                noisy_feature = self.feature_pass_channel(feature, channel_snr)
            else:
                noisy_feature = feature

        elif self.model == 'SwinJSCC_w/_RA' or self.model == 'SwinJSCC_w/_SAandRA':
            feature, mask = self.encoder(input_image, encoder_snr, channel_number, self.model)
            CBR = channel_number / (2 * 3 * 2 ** (self.downsample * 2))
            avg_pwr = torch.sum(feature ** 2) / mask.sum()
            if self.pass_channel:
                noisy_feature = self.feature_pass_channel(feature, channel_snr, avg_pwr)
            else:
                noisy_feature = feature
            noisy_feature = noisy_feature * mask
        else:
            raise ValueError('Unknown model: {}'.format(self.model))

        recon_image = self.decoder(noisy_feature, decoder_snr, self.model)
        recon_clamped = recon_image.clamp(0., 1.)
        mse = self.squared_difference(input_image * 255., recon_clamped * 255.)
        loss_G = self.distortion_loss.forward(input_image, recon_clamped)
        self.last_snreq_penalty = self.compute_latest_snreq_penalty()
        return recon_image, CBR, avg_snr, mse.mean(), loss_G.mean()

    def _collect_caq_statistics_pairs(self):
        snr_values, a_values, b_values = [], [], []
        for module in self.modules():
            if (hasattr(module, 'last_caq_a') and module.last_caq_a is not None
                    and hasattr(module, 'last_caq_snr') and module.last_caq_snr is not None):
                snr_values.append(module.last_caq_snr.reshape(-1))
                a_values.append(module.last_caq_a.mean(dim=1).reshape(-1))
                b_values.append(module.last_caq_b.mean(dim=1).reshape(-1))
        if len(snr_values) == 0:
            device = next(self.parameters()).device
            empty = torch.empty(0, device=device)
            return empty, empty, empty
        return torch.cat(snr_values), torch.cat(a_values), torch.cat(b_values)

    def _collect_caq_statistics(self):
        # Legacy grid-penalty collector kept for forward_multi_snr.
        a_values, b_values = [], []
        for module in self.modules():
            if hasattr(module, 'last_caq_a') and module.last_caq_a is not None:
                a_values.append(module.last_caq_a.mean())
                b_values.append(module.last_caq_b.mean())
        if len(a_values) == 0:
            device = next(self.parameters()).device
            empty = torch.empty(0, device=device)
            return empty, empty
        return torch.stack(a_values), torch.stack(b_values)

    @staticmethod
    def _pearson_corr_columns(x, y, eps=1e-8):
        x = x.float().flatten()
        y = y.float()
        if y.dim() == 1:
            y = y.view(-1, 1)
        if x.numel() < 2 or y.size(0) < 2:
            return torch.zeros((y.size(1),), device=y.device, dtype=y.dtype)
        if torch.var(x) < eps:
            return torch.zeros((y.size(1),), device=y.device, dtype=y.dtype)
        x = x.view(-1, 1)
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)
        numerator = torch.sum(x * y, dim=0)
        denominator = torch.sqrt(torch.sum(x * x, dim=0) + eps) * torch.sqrt(torch.sum(y * y, dim=0) + eps)
        return numerator / denominator

    def snreq_penalty(self, snr_tensor, a_tensor, b_tensor):
        if self.snr_eq_mode not in ('caq', 'emb_caq'):
            return torch.zeros((), device=snr_tensor.device)
        if a_tensor.numel() == 0 or b_tensor.numel() == 0:
            return torch.zeros((), device=snr_tensor.device)
        corr_a = self._pearson_corr_columns(snr_tensor, a_tensor)
        corr_b = self._pearson_corr_columns(snr_tensor, b_tensor)
        return F.relu(corr_a).mean() + F.relu(corr_b).mean()

    def compute_latest_snreq_penalty(self):
        snr_values, a_values, b_values = self._collect_caq_statistics_pairs()
        if snr_values.numel() < 2:
            device = next(self.parameters()).device
            return torch.zeros((), device=device)
        return self.snreq_penalty(snr_values, a_values, b_values)

    def forward_multi_snr(self, input_image, snr_list=None, given_rate=None):
        """Old multi-SNR-grid loss. Useful for ablation, not recommended for fading_multi."""
        if snr_list is None:
            snr_list = self.multiple_snr
        if given_rate is None:
            given_rate = choice(self.channel_number)

        losses, mses, cbrs = [], [], []
        a_stats, b_stats, used_snrs = [], [], []
        last_recon = None
        for snr in snr_list:
            recon_image, CBR, chan_param, mse, loss_G = self.forward(input_image, given_SNR=snr, given_rate=given_rate)
            penalty_single = self.last_snreq_penalty if self.last_snreq_penalty is not None else torch.zeros((), device=input_image.device)
            losses.append(loss_G + self.snreq_lambda * penalty_single)
            mses.append(mse)
            cbrs.append(torch.as_tensor(CBR, device=input_image.device, dtype=torch.float32))
            last_recon = recon_image
            if self.snr_eq_mode in ('caq', 'emb_caq'):
                a_vec, b_vec = self._collect_caq_statistics()
                if a_vec.numel() > 0:
                    a_stats.append(a_vec)
                    b_stats.append(b_vec)
                used_snrs.append(_mean_float(chan_param))

        distortion = torch.stack(losses).mean()
        mse_mean = torch.stack(mses).mean()
        cbr_mean = torch.stack(cbrs).mean().item()
        snr_mean = float(sum([float(x) for x in snr_list])) / float(len(snr_list))

        if self.snr_eq_mode in ('caq', 'emb_caq') and len(a_stats) >= 2:
            snr_tensor = torch.tensor(used_snrs, device=input_image.device, dtype=torch.float32)
            a_tensor = torch.stack(a_stats)
            b_tensor = torch.stack(b_stats)
            grid_penalty = self.snreq_penalty(snr_tensor, a_tensor, b_tensor)
        else:
            grid_penalty = torch.zeros((), device=input_image.device)

        total_loss = distortion + self.snreq_lambda * grid_penalty
        return {
            'recon_image': last_recon,
            'CBR': cbr_mean,
            'SNR': snr_mean,
            'mse': mse_mean,
            'loss': total_loss,
            'distortion': distortion.detach(),
            'snreq_penalty': grid_penalty.detach(),
        }
