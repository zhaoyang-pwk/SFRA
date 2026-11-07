#!/usr/bin/env bash
# Evaluate a trained checkpoint on Kodak
python main.py \
  --trainset DIV2K --testset kodak --test-data-dir data/kodak \
  --C 96 --multiple-snr 0,5,10,15,20 \
  --snr-sample-mode fading_multi --fading-blocks 8 \
  --model SwinJSCC_w/o_SAandRA \
  --sqa-mode emb_caq --sqa-scope channel_side2 \
  --sfrc-mode sfrc --sfrc-scope channel_side2 \
  --resume checkpoints/sfra_jscc.model