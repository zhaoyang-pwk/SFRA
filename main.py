import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import copy
import random
import time
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data.datasets import get_loader
from loss.distortion import *
from net.encoder import is_snreq_model, snreq_mode_from_model
from net.network import SwinJSCC
from utils import *


parser = argparse.ArgumentParser(description='SFRA-JSCC: SQA query attention + SFRC frequency residual correction')
parser.add_argument('--training', action='store_true', help='training or testing')
parser.add_argument('--trainset', type=str, default='DIV2K', choices=['CIFAR10', 'DIV2K'], help='train dataset name')
parser.add_argument('--testset', type=str, default='ffhq', choices=['kodak', 'CLIC21', 'ffhq'], help='default HR testset')
parser.add_argument('--train-data-dir', type=str, default='', help='override train data directory, comma-separated or glob; optional')
parser.add_argument('--test-data-dir', type=str, default='', help='override test data directory')
parser.add_argument('--distortion-metric', type=str, default='MSE', choices=['MSE', 'MS-SSIM', 'MSE+LPIPS'])
parser.add_argument('--mse-weight', type=float, default=1.0,
                    help='weight of the MSE term when --distortion-metric is MSE+LPIPS')
parser.add_argument('--lpips-weight', type=float, default=1.0,
                    help='weight of the LPIPS term when --distortion-metric is MSE+LPIPS')
parser.add_argument('--lpips-net', type=str, default='alex', choices=['alex', 'vgg'],
                    help='LPIPS backbone, following the reference implementation')
parser.add_argument('--model', type=str, default='SwinJSCC_w/o_SAandRA', choices=[
    'SwinJSCC_w/o_SAandRA', 'SwinJSCC_w/_SA', 'SwinJSCC_w/_RA', 'SwinJSCC_w/_SAandRA',
    'SNR_EQ_JSCC', 'SNR_EQ_JSCC_w/o_CAQ', 'SNR_EQ_JSCC_w/o_EM', 'SNR_EQ_JSCC_w/o_CA'
])
parser.add_argument('--channel-type', type=str, default='awgn', choices=['none', 'awgn', 'rayleigh'], help='physical channel branch')
parser.add_argument('--power-norm', type=str, default='global', choices=['global', 'sample'], help='scalar-SNR power normalization mode; tensor-SNR uses per-sample normalization')
parser.add_argument('--C', type=str, default='96', help='bottleneck dimension; comma-separated values are allowed for testing')
parser.add_argument('--multiple-snr', type=str, default='0,5,10,15,20', help='comma-separated average SNR grid in dB')
parser.add_argument('--model_size', type=str, default='base', choices=['small', 'base', 'large'])

# SQA switches (mechanism adapted from SNR-EQ-JSCC). These work with model=SwinJSCC_w/o_SAandRA, so you do not need
# to rename the model just to enable the attention adaptation path.
parser.add_argument('--sqa-mode', '--snr-eq-mode', dest='snr_eq_mode', type=str, default='none', choices=['none', 'emb', 'caq', 'emb_caq'], help='SQA mode: SNR embedding and/or channel-adaptive query')
parser.add_argument('--sqa-scope', '--snr-eq-scope', dest='snr_eq_scope', type=str, default='none', choices=['none', 'bottleneck', 'channel_side2', 'all'], help='where to enable SQA attention')
parser.add_argument('--sqa-hidden-dim', '--snr-eq-hidden-dim', dest='snr_eq_hidden_dim', type=int, default=128)
parser.add_argument('--sqa-snr-scale', '--snr-scale', dest='snr_scale', type=float, default=20.0)
parser.add_argument('--sqa-emb-scale', '--snr-emb-scale', dest='snr_emb_scale', type=float, default=0.1, help='small scale to avoid SNR embedding perturbing a trained backbone')
parser.add_argument('--sqa-penalty-lambda', '--snr-eq-penalty-lambda', '--snreq-lambda', dest='snr_eq_penalty_lambda', type=float, default=1e-5)

# SFRC switches. The frequency branch is internal-only: FFT -> selection -> iFFT -> residual fusion,
# so it keeps the transmitted latent shape/CBR unchanged.
parser.add_argument('--sfrc-mode', '--freq-enhance-mode', dest='freq_enhance_mode', type=str, default='none', choices=['none', 'sfrc', 'sfra'],
                    help='SFRC mode; legacy value sfra remains accepted')
parser.add_argument('--sfrc-scope', '--freq-enhance-scope', dest='freq_enhance_scope', type=str, default='none', choices=['none', 'bottleneck', 'channel_side2', 'all'],
                    help='where to enable SFRC inside SFRA blocks')
parser.add_argument('--sfrc-reduction', '--freq-reduction', dest='freq_reduction', type=int, default=4, help='hidden reduction ratio for SFRC SNR gate')
parser.add_argument('--sfrc-snr-min', '--freq-snr-min', dest='freq_snr_min', type=float, default=0.0, help='SNR lower bound for SFRC gate normalization')
parser.add_argument('--sfrc-snr-max', '--freq-snr-max', dest='freq_snr_max', type=float, default=20.0, help='SNR upper bound for SFRC gate normalization')
parser.add_argument('--sfrc-residual-init', '--freq-residual-init', dest='freq_residual_init', type=float, default=1.0,
                    help='SFRC residual scale gamma init; projection is zero-initialized, so the block starts as identity')
parser.add_argument('--sfrc-max-modulation', '--freq-max-modulation', dest='freq_max_modulation', type=float, default=0.5,
                    help='maximum low/high frequency scale offset around 1.0')

# SNR sampling modes.
parser.add_argument('--train-snr-mode', type=str, default='random', choices=['random', 'multi'], help='multi loops over the SNR grid; random uses snr-sample-mode once per batch')
parser.add_argument('--snr-sample-mode', type=str, default='batch_choice', choices=['batch_choice', 'per_image_choice', 'per_image_uniform', 'fading_multi', 'rayleigh_snr_eq'], help='training/eval SNR sampling mode')
parser.add_argument('--train-snr-min', type=float, default=None)
parser.add_argument('--train-snr-max', type=float, default=None)
parser.add_argument('--fading-blocks', '--rayleigh-fading-blocks', dest='fading_blocks', type=int, default=8, help='M instantaneous-SNR blocks per image in fading_multi mode')
parser.add_argument('--fading-min-h', type=float, default=1e-4, help='clamp |h|^2 to avoid -inf instantaneous SNR')
parser.add_argument('--fading-avg-snr-mode', type=str, default='choice', choices=['choice', 'uniform'], help='how to sample average SNR in fading_multi training')
parser.add_argument('--snr-feedback-mode', type=str, default='avg', choices=['avg', 'instant_mean'], help='SNR condition sent to decoder attention')

parser.add_argument('--learning-rate', type=float, default=1e-4)
parser.add_argument('--tot-epoch', type=int, default=400)
parser.add_argument('--eval-freq', '--save-model-freq', dest='eval_freq', type=int, default=20)
parser.add_argument('--early-stop', action='store_true')
parser.add_argument('--early-stop-metric', type=str, default='balanced_score', choices=['mean_psnr', 'worst_psnr', 'low_snr_mean', 'balanced_score', 'mean_msssim'])
parser.add_argument('--early-stop-min-delta', type=float, default=0.005)
parser.add_argument('--early-stop-patience', '--patience', dest='early_stop_patience', type=int, default=10)
parser.add_argument('--early-stop-warmup', type=int, default=0, help='epochs before early stop starts checking')
parser.add_argument('--fixed-test-seed', type=int, default=-1, help='>=0 fixes test channel noise for stable comparisons')

# Teacher anchor for adapting a fixed-SNR checkpoint to multi-SNR training.
# Intended for B0 only: initialize from fixed SNR=10, CBR=1/16, then train No-EQ
# multi-SNR while softly preserving the original 10 dB reconstruction behavior.
parser.add_argument('--teacher-anchor', action='store_true',
                    help='enable frozen fixed-SNR teacher output anchor during training')
parser.add_argument('--teacher-anchor-snr', type=float, default=10.0,
                    help='anchor SNR in dB; use 10 for your fixed-SNR=10 checkpoint')
parser.add_argument('--teacher-anchor-lambda', type=float, default=0.03,
                    help='weight for teacher anchor loss; recommended 0.02-0.05')
parser.add_argument('--teacher-anchor-loss', type=str, default='mse', choices=['mse', 'l1'],
                    help='distance between student and frozen teacher reconstructions')
parser.add_argument('--teacher-anchor-seed', type=int, default=12345,
                    help='seed used to match stochastic fading/AWGN noise between teacher and student; <0 disables')
parser.add_argument('--teacher-anchor-rate', type=int, default=-1,
                    help='channel number used for anchor forward; default uses first --C value')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--resume', '--pretrained', dest='resume', type=str, default='', help='checkpoint path for warm-start/test')
parser.add_argument('--strict-load', action='store_true')
parser.add_argument('--freeze-backbone', action='store_true', help='use the existing adaptation-only training policy')
parser.add_argument('--partial-finetune', action='store_true', help='train adaptation parameters + encoder last two layers + decoder first two layers')
args = parser.parse_args()

# Keep legacy config values so old commands and checkpoints use the same graph.
if args.freq_enhance_mode == 'sfrc':
    args.freq_enhance_mode = 'sfra'

# Legacy SNR_EQ_JSCC model names still imply SNR-EQ even if --snr-eq-mode is omitted.
if is_snreq_model(args.model) and args.snr_eq_mode == 'none':
    args.snr_eq_mode = snreq_mode_from_model(args.model)
    if args.snr_eq_scope == 'none':
        args.snr_eq_scope = 'all'


# Filled after loading the fixed-SNR checkpoint when --teacher-anchor is enabled.
teacher_model = None


class config():
    seed = args.seed
    pass_channel = True
    CUDA = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    norm = False

    print_step = 100
    plot_step = 10000
    filename = datetime.now().__str__()[:-7].replace(':', '-')
    workdir = './history/{}'.format(filename)
    log = workdir + '/Log_{}.log'.format(filename)
    samples = workdir + '/samples'
    models = workdir + '/models'
    logger = None

    normalize = False
    learning_rate = args.learning_rate
    distortion_metric = args.distortion_metric
    mse_weight = args.mse_weight
    lpips_weight = args.lpips_weight
    lpips_net = args.lpips_net
    tot_epoch = args.tot_epoch
    save_model_freq = args.eval_freq

    if args.trainset == 'CIFAR10':
        image_dims = (3, 32, 32)
        train_data_dir = args.train_data_dir or "data/cifar10"
        test_data_dir = args.test_data_dir or "data/cifar10"
        batch_size = 128
        downsample = 2
        channel_number = int(args.C.split(',')[0])
        encoder_kwargs = dict(
            model=args.model, img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 256], depths=[2, 4], num_heads=[4, 8], C=channel_number,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm,
            patch_norm=True, snr_eq_mode=args.snr_eq_mode, snr_eq_scope=args.snr_eq_scope,
            snr_eq_hidden_dim=args.snr_eq_hidden_dim, snr_scale=args.snr_scale,
            snr_emb_scale=args.snr_emb_scale,
            freq_enhance_mode=args.freq_enhance_mode, freq_enhance_scope=args.freq_enhance_scope,
            freq_reduction=args.freq_reduction, freq_snr_min=args.freq_snr_min,
            freq_snr_max=args.freq_snr_max, freq_residual_init=args.freq_residual_init,
            freq_max_modulation=args.freq_max_modulation,
        )
        decoder_kwargs = dict(
            model=args.model, img_size=(image_dims[1], image_dims[2]), embed_dims=[256, 128],
            depths=[4, 2], num_heads=[8, 4], C=channel_number, window_size=2, mlp_ratio=4.,
            qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm, patch_norm=True,
            snr_eq_mode=args.snr_eq_mode, snr_eq_scope=args.snr_eq_scope,
            snr_eq_hidden_dim=args.snr_eq_hidden_dim, snr_scale=args.snr_scale,
            snr_emb_scale=args.snr_emb_scale,
            freq_enhance_mode=args.freq_enhance_mode, freq_enhance_scope=args.freq_enhance_scope,
            freq_reduction=args.freq_reduction, freq_snr_min=args.freq_snr_min,
            freq_snr_max=args.freq_snr_max, freq_residual_init=args.freq_residual_init,
            freq_max_modulation=args.freq_max_modulation,
        )
    elif args.trainset == 'DIV2K':
        image_dims = (3, 256, 256)
        base_path = "data/DIV2K"
        if args.test_data_dir:
            test_data_dir = [args.test_data_dir]
        elif args.testset == 'kodak':
            test_data_dir = ["data/kodak"]
        elif args.testset == 'CLIC21':
            test_data_dir = ["data/clic2021"]
        elif args.testset == 'ffhq':
            test_data_dir = ["data/ffhq"]
        else:
            test_data_dir = [args.test_data_dir]

        if args.train_data_dir:
            train_data_dir = [x for x in args.train_data_dir.split(',') if x]
        else:
            train_data_dir = [
                base_path + '/clic2020/**', base_path + '/clic2021/train', base_path + '/clic2021/valid',
                base_path + '/clic2022/val', base_path + '/DIV2K_train_HR', base_path + '/DIV2K_valid_HR'
            ]
        batch_size = 4
        downsample = 4
        if args.model == 'SwinJSCC_w/o_SAandRA' or args.model == 'SwinJSCC_w/_SA' or is_snreq_model(args.model):
            channel_number = int(args.C.split(',')[0])
        else:
            channel_number = None

        if args.model_size == 'small':
            encoder_depths = [2, 2, 2, 2]
            decoder_depths = [2, 2, 2, 2]
        elif args.model_size == 'base':
            encoder_depths = [2, 2, 6, 2]
            decoder_depths = [2, 6, 2, 2]
        elif args.model_size == 'large':
            encoder_depths = [2, 2, 18, 2]
            decoder_depths = [2, 18, 2, 2]
        else:
            raise ValueError('Unknown model_size {}'.format(args.model_size))

        encoder_kwargs = dict(
            model=args.model, img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 192, 256, 320], depths=encoder_depths, num_heads=[4, 6, 8, 10],
            C=channel_number, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True, snr_eq_mode=args.snr_eq_mode,
            snr_eq_scope=args.snr_eq_scope, snr_eq_hidden_dim=args.snr_eq_hidden_dim,
            snr_scale=args.snr_scale, snr_emb_scale=args.snr_emb_scale,
            freq_enhance_mode=args.freq_enhance_mode, freq_enhance_scope=args.freq_enhance_scope,
            freq_reduction=args.freq_reduction, freq_snr_min=args.freq_snr_min,
            freq_snr_max=args.freq_snr_max, freq_residual_init=args.freq_residual_init,
            freq_max_modulation=args.freq_max_modulation,
        )
        decoder_kwargs = dict(
            model=args.model, img_size=(image_dims[1], image_dims[2]), embed_dims=[320, 256, 192, 128],
            depths=decoder_depths, num_heads=[10, 8, 6, 4], C=channel_number, window_size=8,
            mlp_ratio=4., qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm, patch_norm=True,
            snr_eq_mode=args.snr_eq_mode, snr_eq_scope=args.snr_eq_scope,
            snr_eq_hidden_dim=args.snr_eq_hidden_dim, snr_scale=args.snr_scale,
            snr_emb_scale=args.snr_emb_scale,
            freq_enhance_mode=args.freq_enhance_mode, freq_enhance_scope=args.freq_enhance_scope,
            freq_reduction=args.freq_reduction, freq_snr_min=args.freq_snr_min,
            freq_snr_max=args.freq_snr_max, freq_residual_init=args.freq_residual_init,
            freq_max_modulation=args.freq_max_modulation,
        )


if args.trainset == 'CIFAR10':
    CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).to(config.device)
else:
    CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).to(config.device)


def parse_float_list(text):
    return [float(i) for i in text.split(',') if i != '']


def parse_int_list(text):
    return [int(i) for i in text.split(',') if i != '']


def calc_psnr_from_mse(mse):
    if torch.is_tensor(mse):
        return 10 * (torch.log(255. * 255. / mse) / np.log(10))
    return 10 * np.log10(255. * 255. / mse)


def load_weights(model_path):
    checkpoint = torch.load(model_path, map_location=config.device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']

    # strict=False still errors on same-name keys with different shapes.
    # Skip shape-mismatched buffers such as Swin attn_mask; they are regenerated by the current model.
    if not args.strict_load:
        current_state = net.state_dict()
        filtered = {}
        skipped = []
        for k, v in checkpoint.items():
            kk = k[7:] if k.startswith('module.') else k
            if kk in current_state and hasattr(v, 'shape') and current_state[kk].shape != v.shape:
                skipped.append((kk, tuple(v.shape), tuple(current_state[kk].shape)))
                continue
            filtered[kk] = v
        if skipped:
            logger.warning('Skipped {} shape-mismatched checkpoint keys: {}'.format(len(skipped), skipped[:30]))
        checkpoint = filtered

    msg = net.load_state_dict(checkpoint, strict=args.strict_load)
    if not args.strict_load:
        logger.info('Loaded checkpoint non-strictly: {}'.format(msg))
    del checkpoint


def set_trainable_policy(model):
    if not args.freeze_backbone and not args.partial_finetune:
        logger.info('Training policy: full fine-tune, train all parameters.')
        return
    for _, p in model.named_parameters():
        p.requires_grad = False
    snreq_keys = ('snr_embed', 'caq_a', 'caq_b')
    sfra_keys = ('freq_adapter', 'norm_freq')
    partial_keys = ('encoder.layers.2', 'encoder.layers.3', 'decoder.layers.0', 'decoder.layers.1',
                    'encoder.norm', 'encoder.head_list', 'decoder.head_list')
    enabled = []
    for name, p in model.named_parameters():
        train_this = any(k in name for k in snreq_keys) or (args.freq_enhance_mode == 'sfra' and any(k in name for k in sfra_keys))
        if args.partial_finetune:
            train_this = train_this or any(k in name for k in partial_keys)
        if train_this:
            p.requires_grad = True
            enabled.append(name)
    logger.info('Training policy: {}'.format('partial fine-tune' if args.partial_finetune else 'freeze backbone'))
    logger.info('Trainable parameter count: {} / {}'.format(
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        sum(p.numel() for p in model.parameters())
    ))
    logger.info('Trainable parameter names sample: {}'.format(enabled[:80]))


@contextmanager
def fixed_seed_context(seed):
    if seed is None or seed < 0:
        yield
        return
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    seed_torch(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if torch.cuda.is_available() and cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)



def build_frozen_teacher_from_current_model(model):
    """Copy the already-loaded fixed-SNR student as a frozen teacher."""
    teacher = copy.deepcopy(model)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def teacher_anchor_forward_loss(input_image):
    """Softly keep the current model close to the fixed-SNR teacher at anchor SNR.

    This is designed for B0 only:
        F0 fixed SNR=10, CBR=1/16 -> B0 No-EQ multi-SNR adaptation.

    It runs one extra student forward and one frozen teacher forward at 10 dB,
    using the same random seed so fading_multi instantaneous SNR and AWGN noise
    are matched as closely as possible.
    """
    if teacher_model is None or args.teacher_anchor_lambda <= 0:
        return torch.zeros((), device=config.device)

    anchor_rate = args.teacher_anchor_rate
    if anchor_rate is None or anchor_rate <= 0:
        anchor_rate = int(args.C.split(',')[0])

    seed = args.teacher_anchor_seed if args.teacher_anchor_seed is not None and args.teacher_anchor_seed >= 0 else -1

    # Student anchor forward keeps gradients.
    with fixed_seed_context(seed):
        student_recon, _, _, _, _ = net(input_image, args.teacher_anchor_snr, anchor_rate)

    # Teacher anchor forward is frozen and graph-free.
    with torch.no_grad():
        with fixed_seed_context(seed):
            teacher_recon, _, _, _, _ = teacher_model(input_image, args.teacher_anchor_snr, anchor_rate)

    student_recon = student_recon.clamp(0., 1.)
    teacher_recon = teacher_recon.clamp(0., 1.).detach()
    if args.teacher_anchor_loss == 'l1':
        raw = F.l1_loss(student_recon, teacher_recon)
    else:
        raw = F.mse_loss(student_recon, teacher_recon)
    return float(args.teacher_anchor_lambda) * raw

def train_one_epoch(epoch):
    net.train()
    elapsed, losses, psnrs, msssims, cbrs, snrs, penalties, anchors = [AverageMeter() for _ in range(8)]
    metrics = [elapsed, losses, psnrs, msssims, cbrs, snrs, penalties, anchors]
    global global_step

    for batch_idx, batch in enumerate(train_loader):
        start_time = time.time()
        global_step += 1
        if args.trainset == 'CIFAR10':
            input_image, _ = batch
        else:
            input_image = batch
        input_image = input_image.to(config.device)

        use_grid_multi = args.train_snr_mode == 'multi' and args.snr_sample_mode not in ('fading_multi', 'rayleigh_snr_eq') and len(net.multiple_snr) > 1
        if use_grid_multi:
            out = net.forward_multi_snr(input_image, snr_list=net.multiple_snr)
            recon_image = out['recon_image']
            CBR = out['CBR']
            SNR = out['SNR']
            mse = out['mse']
            loss = out['loss']
            snreq_penalty = out['snreq_penalty']
        else:
            recon_image, CBR, SNR, mse, loss_G = net(input_image)
            snreq_penalty = net.last_snreq_penalty if net.last_snreq_penalty is not None else torch.zeros((), device=config.device)
            loss = loss_G + net.snreq_lambda * snreq_penalty

        loss_anchor = torch.zeros((), device=config.device)
        if teacher_model is not None and args.teacher_anchor_lambda > 0:
            loss_anchor = teacher_anchor_forward_loss(input_image)
            loss = loss + loss_anchor

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        elapsed.update(time.time() - start_time)
        losses.update(float(loss.item()))
        cbrs.update(float(CBR))
        snrs.update(float(SNR.detach().float().mean().item()) if torch.is_tensor(SNR) else float(SNR))
        penalties.update(float(snreq_penalty.detach().item()))
        anchors.update(float(loss_anchor.detach().item()))
        if mse.item() > 0:
            psnr = calc_psnr_from_mse(mse)
            psnrs.update(psnr.item())
            msssim = 1 - CalcuSSIM(input_image, recon_image.clamp(0., 1.)).mean().item()
            msssims.update(msssim)

        if (global_step % config.print_step) == 0:
            process = (global_step % len(train_loader)) / len(train_loader) * 100.0
            log = (' | '.join([
                f'Epoch {epoch}', f'Step [{global_step % len(train_loader)}/{len(train_loader)}={process:.2f}%]',
                f'Time {elapsed.val:.3f}', f'Loss {losses.val:.6f} ({losses.avg:.6f})',
                f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})', f'SNR {snrs.val:.2f} ({snrs.avg:.2f})',
                f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})', f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                f'CAQ_penalty {penalties.val:.6f} ({penalties.avg:.6f})',
                f'Anchor {anchors.val:.6f} ({anchors.avg:.6f})', f'Lr {cur_lr}',
            ]))
            logger.info(log)
            for i in metrics:
                i.clear()

    for i in metrics:
        i.clear()


def test():
    net.eval()
    multiple_snr = parse_float_list(args.multiple_snr)
    channel_number = parse_int_list(args.C)
    results_snr = np.zeros((len(multiple_snr), len(channel_number)))
    results_cbr = np.zeros((len(multiple_snr), len(channel_number)))
    results_psnr = np.zeros((len(multiple_snr), len(channel_number)))
    results_msssim = np.zeros((len(multiple_snr), len(channel_number)))

    for i, SNR in enumerate(multiple_snr):
        for j, rate in enumerate(channel_number):
            elapsed, psnrs, msssims, snrs, cbrs = [AverageMeter() for _ in range(5)]
            seed = args.fixed_test_seed + i * 1009 + j * 9176 if args.fixed_test_seed >= 0 else -1
            with fixed_seed_context(seed), torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    if args.trainset == 'CIFAR10':
                        input_image, _ = batch
                    else:
                        input_image, _ = batch
                    start_time = time.time()
                    input_image = input_image.to(config.device)
                    recon_image, CBR, used_SNR, mse, loss_G = net(input_image, SNR, rate)
                    elapsed.update(time.time() - start_time)
                    cbrs.update(float(CBR))
                    snrs.update(float(used_SNR.detach().float().mean().item()) if torch.is_tensor(used_SNR) else float(used_SNR))
                    if mse.item() > 0:
                        psnr = calc_psnr_from_mse(mse)
                        psnrs.update(psnr.item())
                        msssim = 1 - CalcuSSIM(input_image, recon_image.clamp(0., 1.)).mean().item()
                        msssims.update(msssim)

            results_snr[i, j] = snrs.avg
            results_cbr[i, j] = cbrs.avg
            results_psnr[i, j] = psnrs.avg
            results_msssim[i, j] = msssims.avg
            logger.info('Eval | SNR {} | C {} | CBR {:.4f} | PSNR {:.4f} | MS-SSIM {:.6f}'.format(
                SNR, rate, cbrs.avg, psnrs.avg, msssims.avg
            ))

    mean_psnr = float(np.mean(results_psnr))
    worst_psnr = float(np.min(results_psnr))
    order = np.argsort(multiple_snr)
    low_count = min(2, len(order))
    low_snr_mean = float(np.mean(results_psnr[order[:low_count], :]))
    balanced_score = mean_psnr + 0.3 * worst_psnr
    mean_msssim = float(np.mean(results_msssim))

    print("SNR: {}".format(results_snr.tolist()))
    print("CBR: {}".format(results_cbr.tolist()))
    print("PSNR: {}".format(results_psnr.tolist()))
    print("MS-SSIM: {}".format(results_msssim.tolist()))
    print("mean_psnr: {:.6f}, worst_psnr: {:.6f}, low_snr_mean: {:.6f}, balanced_score: {:.6f}, mean_msssim: {:.6f}".format(
        mean_psnr, worst_psnr, low_snr_mean, balanced_score, mean_msssim
    ))
    print("Finish Test!")
    return {
        'mean_psnr': mean_psnr, 'worst_psnr': worst_psnr, 'low_snr_mean': low_snr_mean,
        'balanced_score': balanced_score, 'mean_msssim': mean_msssim,
        'results_psnr': results_psnr, 'results_msssim': results_msssim,
    }


if __name__ == '__main__':
    seed_torch(config.seed)
    logger = logger_configuration(config, save_log=True)
    logger.info(config.__dict__)
    logger.info('Args: {}'.format(vars(args)))
    torch.manual_seed(seed=config.seed)
    net = SwinJSCC(args, config).to(config.device)
    if args.resume:
        load_weights(args.resume)

    if args.training and args.teacher_anchor:
        if not args.resume:
            logger.warning('[TeacherAnchor] --teacher-anchor is enabled without --pretrained/--resume. '
                           'Teacher will copy a randomly initialized model, which is usually not intended.')
        if args.snr_eq_mode != 'none':
            logger.warning('[TeacherAnchor] enabled while snr_eq_mode is not none. '
                           'Recommended use is B0 No-EQ adaptation only; disable it for E1/E2 unless intentional.')
        teacher_model = build_frozen_teacher_from_current_model(net)
        logger.info('[TeacherAnchor] Frozen teacher copied after checkpoint loading. '
                    'snr={:.2f}, lambda={}, loss={}, rate={}'.format(
                        args.teacher_anchor_snr, args.teacher_anchor_lambda,
                        args.teacher_anchor_loss,
                        args.teacher_anchor_rate if args.teacher_anchor_rate > 0 else int(args.C.split(',')[0])
                    ))

    set_trainable_policy(net)
    model_params = [{'params': [p for p in net.parameters() if p.requires_grad], 'lr': config.learning_rate}]
    if len(model_params[0]['params']) == 0:
        raise RuntimeError('No trainable parameters. Check --freeze-backbone/--partial-finetune and SQA/SFRC mode/scope.')
    train_loader, test_loader = get_loader(args, config)
    cur_lr = config.learning_rate
    optimizer = optim.Adam(model_params, lr=cur_lr)
    global_step = 0

    if args.training:
        best_values = {
            'mean_psnr': -float('inf'), 'worst_psnr': -float('inf'), 'low_snr_mean': -float('inf'),
            'balanced_score': -float('inf'), 'mean_msssim': -float('inf')
        }
        no_improve_rounds = 0
        for epoch in range(0, config.tot_epoch):
            train_one_epoch(epoch)
            if (epoch + 1) % config.save_model_freq == 0:
                eval_metrics = test()

                # Save only the model that improves the monitored objective.
                # This avoids filling disk with EPxxxx.model/latest.model checkpoints.
                monitored = eval_metrics[args.early_stop_metric]
                best_key = '_monitored_best'
                if monitored > best_values.get(best_key, -float('inf')) + args.early_stop_min_delta:
                    best_values[best_key] = monitored
                    no_improve_rounds = 0
                    save_name = 'best_{}.model'.format(args.early_stop_metric)
                    save_path = config.models + '/' + save_name
                    save_model(net, save_path=save_path)
                    logger.info('Updated {}: {:.6f}'.format(save_name, monitored))
                else:
                    no_improve_rounds += 1
                    logger.info('{} did not improve for {} eval rounds. Best: {:.6f}, current: {:.6f}'.format(
                        args.early_stop_metric, no_improve_rounds,
                        best_values.get(best_key, -float('inf')), monitored
                    ))

                if (args.early_stop and (epoch + 1) >= args.early_stop_warmup
                        and no_improve_rounds >= args.early_stop_patience):
                    logger.info('Early stopping triggered at epoch {}.'.format(epoch + 1))
                    break
    else:
        test()
