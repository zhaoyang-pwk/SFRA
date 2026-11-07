import numpy as np
import torch
import torch.nn as nn


class Channel(nn.Module):
    """Complex AWGN / Rayleigh / SNR-equivalent fading channel.

    Backward-compatible scalar-SNR path is kept for the original SwinJSCC
    experiments. The new tensor-SNR path supports:
      * snr_db: scalar          -> one SNR for the whole batch
      * snr_db: [B]             -> one SNR per image
      * snr_db: [B, M]          -> M instantaneous SNR blocks per image

    The [B, M] mode is the key for SNR-EQ-style fading without maintaining
    separate AWGN/Rayleigh branches: Rayleigh fading is first converted to
    instantaneous SNRs, then the latent symbols are corrupted block-by-block by
    AWGN at those instantaneous SNRs.
    """

    def __init__(self, args, config):
        super().__init__()
        self.config = config
        self.chan_type = getattr(args, 'channel_type', 'awgn')
        self.device = config.device
        self.power_norm = getattr(args, 'power_norm', 'global')
        if config.logger:
            config.logger.info('【Channel】: Built {} channel, SNR {} dB, power_norm={}.'.format(
                self.chan_type, args.multiple_snr, self.power_norm))

    @staticmethod
    def _as_snr_tensor(snr_db, device, dtype=torch.float32):
        if torch.is_tensor(snr_db):
            return snr_db.to(device=device, dtype=dtype)
        return torch.tensor(snr_db, device=device, dtype=dtype)

    @staticmethod
    def _snr_to_sigma(snr_db):
        return torch.sqrt(1.0 / (2.0 * torch.pow(10.0, snr_db / 10.0)))

    def gaussian_noise_layer(self, input_layer, std, name=None):
        device = input_layer.device
        if not torch.is_tensor(std):
            std = torch.tensor(std, device=device, dtype=input_layer.real.dtype)
        std = std.to(device=device, dtype=input_layer.real.dtype)
        while std.dim() < input_layer.dim():
            std = std.unsqueeze(-1)
        noise_real = torch.normal(mean=0.0, std=1.0, size=input_layer.shape, device=device) * std
        noise_imag = torch.normal(mean=0.0, std=1.0, size=input_layer.shape, device=device) * std
        noise = noise_real + 1j * noise_imag
        return input_layer + noise

    def rayleigh_noise_layer(self, input_layer, std, name=None):
        device = input_layer.device
        if not torch.is_tensor(std):
            std = torch.tensor(std, device=device, dtype=input_layer.real.dtype)
        std = std.to(device=device, dtype=input_layer.real.dtype)
        while std.dim() < input_layer.dim():
            std = std.unsqueeze(-1)
        noise_real = torch.normal(mean=0.0, std=1.0, size=input_layer.shape, device=device) * std
        noise_imag = torch.normal(mean=0.0, std=1.0, size=input_layer.shape, device=device) * std
        noise = noise_real + 1j * noise_imag
        h = torch.sqrt(torch.randn(input_layer.shape, device=device) ** 2
                       + torch.randn(input_layer.shape, device=device) ** 2) / np.sqrt(2.0)
        return input_layer * h + noise

    def complex_normalize(self, x, power=1.0, mode=None):
        """Normalize real latent to unit complex power.

        mode='global' matches the original implementation closely.
        mode='sample' normalizes each image independently, which is used by the
        new blockwise instantaneous-SNR path.
        """
        mode = mode or self.power_norm
        if mode == 'sample' and x.dim() >= 2:
            dims = tuple(range(1, x.dim()))
            pwr = torch.mean(x ** 2, dim=dims, keepdim=True) * 2.0
        else:
            pwr = torch.mean(x ** 2) * 2.0
        out = np.sqrt(power) * x / torch.sqrt(pwr + 1e-12)
        return out, pwr

    @staticmethod
    def _real_to_complex_per_sample(x):
        B = x.shape[0]
        flat = x.reshape(B, -1)
        L = flat.size(1)
        if L % 2 != 0:
            flat = torch.cat([flat, torch.zeros(B, 1, device=x.device, dtype=x.dtype)], dim=1)
            L = L + 1
        half = L // 2
        z = flat[:, :half] + 1j * flat[:, half:]
        return z, flat.size(1), x.shape

    @staticmethod
    def _complex_to_real_shape(z, original_flat_len, original_shape):
        real_flat = torch.cat([torch.real(z), torch.imag(z)], dim=1)
        real_flat = real_flat[:, :original_flat_len]
        return real_flat.reshape(original_shape)

    def _apply_blockwise_awgn(self, z, snr_db):
        """Apply AWGN with per-image or per-block SNR to complex symbols z [B,L]."""
        B, L = z.shape
        snr = self._as_snr_tensor(snr_db, z.device, z.real.dtype)
        if snr.dim() == 0:
            sigma = self._snr_to_sigma(snr).view(1, 1).expand(B, L)
        elif snr.dim() == 1:
            if snr.numel() == 1:
                sigma = self._snr_to_sigma(snr.view(1)).view(1, 1).expand(B, L)
            elif snr.numel() == B:
                sigma = self._snr_to_sigma(snr).view(B, 1).expand(B, L)
            else:
                sigma = self._snr_to_sigma(snr.mean()).view(1, 1).expand(B, L)
        elif snr.dim() == 2:
            # snr [B,M]. Split every image's complex stream into M contiguous blocks.
            if snr.size(0) != B:
                snr = snr[:1].expand(B, -1)
            M = snr.size(1)
            sigma = torch.empty(B, L, device=z.device, dtype=z.real.dtype)
            edges = torch.linspace(0, L, M + 1, device=z.device).round().long()
            for m in range(M):
                lo = int(edges[m].item())
                hi = int(edges[m + 1].item())
                sigma[:, lo:hi] = self._snr_to_sigma(snr[:, m]).view(B, 1)
        else:
            sigma = self._snr_to_sigma(snr.reshape(-1).mean()).view(1, 1).expand(B, L)

        noise_real = torch.randn_like(z.real) * sigma
        noise_imag = torch.randn_like(z.real) * sigma
        return z + noise_real + 1j * noise_imag

    def _tensor_snr_forward(self, input, snr_db, avg_pwr=False):
        # Per-sample normalization is required for meaningful per-image/block SNR.
        if avg_pwr is not False:
            channel_tx = input / torch.sqrt(avg_pwr * 2.0 + 1e-12)
            restore_pwr = avg_pwr
        else:
            channel_tx, restore_pwr = self.complex_normalize(input, power=1.0, mode='sample')

        z, flat_len, original_shape = self._real_to_complex_per_sample(channel_tx)
        if self.chan_type == 0 or self.chan_type == 'none':
            z_out = z
        elif self.chan_type == 2 or self.chan_type == 'rayleigh':
            # Physical Rayleigh path, kept for comparison/debugging.
            snr = self._as_snr_tensor(snr_db, z.device, z.real.dtype)
            if snr.dim() > 1:
                snr_for_noise = snr.mean(dim=1)
            else:
                snr_for_noise = snr
            sigma = self._snr_to_sigma(snr_for_noise)
            if sigma.dim() == 0:
                sigma = sigma.view(1, 1).expand_as(z.real)
            elif sigma.numel() == z.size(0):
                sigma = sigma.view(z.size(0), 1).expand_as(z.real)
            h = torch.sqrt(torch.randn_like(z.real) ** 2 + torch.randn_like(z.real) ** 2) / np.sqrt(2.0)
            noise = torch.randn_like(z.real) * sigma + 1j * torch.randn_like(z.real) * sigma
            z_out = z * h + noise
        else:
            z_out = self._apply_blockwise_awgn(z, snr_db)

        out = self._complex_to_real_shape(z_out, flat_len, original_shape)
        if avg_pwr is not False:
            return out * torch.sqrt(avg_pwr * 2.0 + 1e-12)
        return out * torch.sqrt(restore_pwr + 1e-12)

    def forward(self, input, chan_param, avg_pwr=False):
        snr = self._as_snr_tensor(chan_param, input.device, input.dtype)
        if snr.numel() > 1:
            return self._tensor_snr_forward(input, snr, avg_pwr=avg_pwr)

        # Original scalar-SNR path, preserved for old baselines.
        snr_scalar = float(snr.reshape(-1)[0].item())
        if avg_pwr:
            channel_tx = input / torch.sqrt(avg_pwr * 2.0 + 1e-12)
        else:
            channel_tx, pwr = self.complex_normalize(input, power=1.0, mode='global')
        input_shape = channel_tx.shape
        channel_in = channel_tx.reshape(-1)
        L = channel_in.shape[0]
        if L % 2 != 0:
            channel_in = torch.cat([channel_in, torch.zeros(1, device=channel_in.device, dtype=channel_in.dtype)], dim=0)
            L = L + 1
        channel_in = channel_in[:L // 2] + channel_in[L // 2:] * 1j
        channel_output = self.complex_forward(channel_in, snr_scalar)
        channel_output = torch.cat([torch.real(channel_output), torch.imag(channel_output)])[:input.numel()]
        channel_output = channel_output.reshape(input_shape)
        if self.chan_type == 1 or self.chan_type == 'awgn':
            noise = (channel_output - channel_tx).detach()
            noise.requires_grad = False
            channel_tx = channel_tx + noise
            if avg_pwr:
                return channel_tx * torch.sqrt(avg_pwr * 2.0 + 1e-12)
            return channel_tx * torch.sqrt(pwr + 1e-12)
        elif self.chan_type == 2 or self.chan_type == 'rayleigh':
            if avg_pwr:
                return channel_output * torch.sqrt(avg_pwr * 2.0 + 1e-12)
            return channel_output * torch.sqrt(pwr + 1e-12)
        elif self.chan_type == 0 or self.chan_type == 'none':
            return input
        return channel_output

    def complex_forward(self, channel_in, chan_param):
        if self.chan_type == 0 or self.chan_type == 'none':
            return channel_in
        if self.chan_type == 1 or self.chan_type == 'awgn':
            sigma = np.sqrt(1.0 / (2.0 * 10 ** (chan_param / 10.0)))
            return self.gaussian_noise_layer(channel_in, std=sigma, name='awgn_chan_noise')
        if self.chan_type == 2 or self.chan_type == 'rayleigh':
            sigma = np.sqrt(1.0 / (2.0 * 10 ** (chan_param / 10.0)))
            return self.rayleigh_noise_layer(channel_in, std=sigma, name='rayleigh_chan_noise')
        return channel_in

    def noiseless_forward(self, channel_in):
        channel_tx, _ = self.complex_normalize(channel_in, power=1.0)
        return channel_tx
