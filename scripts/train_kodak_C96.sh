#!/usr/bin/env bash
# Full SFRA model: DIV2K -> Kodak, C = 96
python main.py --training \
  --trainset DIV2K --testset kodak \
  --train-data-dir data/DIV2K_train_HR --test-data-dir data/kodak \
  --C 96 --multiple-snr 0,5,10,15,20 \
  --snr-sample-mode fading_multi --fading-blocks 8 \
  --model SwinJSCC_w/o_SAandRA \
  --sqa-mode emb_caq --sqa-scope channel_side2 \
  --sfrc-mode sfrc --sfrc-scope channel_side2 \
  --tot-epoch 400 --learning-rate 1e-4