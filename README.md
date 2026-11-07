# SFRA-JSCC

Official code for **“Adaptive Wireless Image Semantic Transmission over Fast Fading Channels”**.

SFRA-JSCC augments a Swin-based joint source–channel coding backbone with a
**Spatial Query Attention (SQA)** module and a **Spatial–Frequency Residual
Correction (SFRC)** module, so that the latent semantic representation is
modulated by the channel-state condition before transmission and restored
according to its degradation after transmission.

* **SQA** injects the channel condition into the query branch of window
  attention, so that the semantic association inside each window follows the
  current channel quality.
* **SFRC** transforms the latent feature into the frequency domain and applies a
  bounded gain to the low- and high-frequency components separately.

Both modules operate on the latent features only; the number of transmitted
channel symbols is unchanged.

## Repository structure

```
.
├── main.py               # training / evaluation entry (PSNR, MS-SSIM)
├── eval_lpips.py         # same entry with LPIPS reported alongside
├── utils.py              # meters, logging helpers
├── data/datasets.py      # DIV2K / Kodak / CIFAR-10 loaders
├── loss/distortion.py    # MSE, MS-SSIM and MSE+LPIPS distortion terms
├── net/
│   ├── modules.py        # SQA, SFRC and the SFRA block
│   ├── encoder.py        # hierarchical semantic encoder
│   ├── decoder.py        # hierarchical semantic decoder
│   ├── network.py        # SwinJSCC backbone (w/o SA & RA)
│   └── channel.py        # AWGN and block-fading equivalent channel
├── scripts/              # example training / evaluation commands
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone <this-repo>
cd SFRA-JSCC
pip install -r requirements.txt
```

Python 3.8+ and PyTorch 1.10+ are recommended. A single GPU is enough for
training at `C = 96` on DIV2K.

## Data

| Dataset | Role | Expected layout |
|---|---|---|
| DIV2K | training | `data/DIV2K_train_HR/*.png` |
| Kodak | high-resolution test | `data/kodak/*.png` |
| CIFAR-10 | low-resolution test | `data/cifar10/test/...` |

Pass the paths with `--train-data-dir` and `--test-data-dir`.

## Training

Full SFRA model on DIV2K → Kodak at `C = 96`:

```bash
python main.py --training \
  --trainset DIV2K --testset kodak \
  --train-data-dir data/DIV2K_train_HR --test-data-dir data/kodak \
  --C 96 --multiple-snr 0,5,10,15,20 \
  --snr-sample-mode fading_multi --fading-blocks 8 \
  --model SwinJSCC_w/o_SAandRA \
  --sqa-mode emb_caq --sqa-scope channel_side2 \
  --sfrc-mode sfrc --sfrc-scope channel_side2 \
  --tot-epoch 400 --learning-rate 1e-4
```

For the low-resolution setting use `--trainset CIFAR10 --testset CIFAR10 --C 32`.

## Evaluation

```bash
python main.py \
  --trainset DIV2K --testset kodak --test-data-dir data/kodak \
  --C 96 --multiple-snr 0,5,10,15,20 \
  --snr-sample-mode fading_multi --fading-blocks 8 \
  --model SwinJSCC_w/o_SAandRA \
  --sqa-mode emb_caq --sqa-scope channel_side2 \
  --sfrc-mode sfrc --sfrc-scope channel_side2 \
  --resume checkpoints/sfra_jscc.model
```

Use `eval_lpips.py` for the same protocol with LPIPS reported alongside PSNR and
MS-SSIM.

## Ablation switches

| Configuration | `--sqa-mode` / `--sqa-scope` | `--sfrc-mode` / `--sfrc-scope` |
|---|---|---|
| Full SFRA | `emb_caq` / `channel_side2` | `sfrc` / `channel_side2` |
| w/o SQA | `none` / `none` | `sfrc` / `channel_side2` |
| w/o SFRC | `emb_caq` / `channel_side2` | `none` / `none` |
| Backbone only | `none` / `none` | `none` / `none` |

Setting both switches to `none` reproduces the SwinJSCC backbone without spatial
adaptation and rate adaptation.

## Block-fading channel

`--snr-sample-mode fading_multi --fading-blocks M` splits the symbol sequence of
one image into `M` blocks, each with an independent Rayleigh gain, while the
channel condition used by SQA/SFRC remains the image-level average SNR. This is
the configuration used throughout the paper (`M = 8`).

## Training objective

The distortion used for training follows the paper,

```
d = lambda_mse * MSE + lambda_lpips * LPIPS
```

with `lambda_mse = lambda_lpips = 1` by default. Enable it with
`--distortion-metric MSE+LPIPS`; the two weights are exposed as `--mse-weight`
and `--lpips-weight`, and the LPIPS backbone can be switched with `--lpips-net
{alex, vgg}`.

The MSE term is computed on the `[-1, 1]` tensors produced by the network, so it
equals four times the MSE on `[0, 1]` pixels. Pass `--mse-weight 0.25` if you
want the exact `[0, 1]` definition used in the paper when reporting.

## Code availability

The code is available at:

```
https://github.com/zhaoyang-pwk/SFRA
```

The repository contains the model, the training/evaluation pipeline and the
ablation switches needed to reproduce the results in the paper. Trained weights
are not released; retraining with the commands above reproduces the reported
numbers under the same data split and random seeds.


## Acknowledgements

The backbone follows
[SwinJSCC](https://github.com/semcomm/SwinJSCC). The idea of embedding the
channel condition into the attention computation is inspired by
[SNR-EQ-JSCC](https://arxiv.org/abs/2501.04732). We thank the authors for
releasing their code.

## Citation

```bibtex
@article{peng2026sfra,
  title   = {Adaptive Wireless Image Semantic Transmission over Fast Fading Channels},
  author  = {Peng, Yi and Pu, Wenkai and Yang, Qingqing and Li, Hui and Wang, Jianming},
  journal = {Journal of Beijing University of Aeronautics and Astronautics},
  year    = {2026}
}
```

## License

MIT License. See [LICENSE](LICENSE).
