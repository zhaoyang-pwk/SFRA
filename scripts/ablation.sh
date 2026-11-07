#!/usr/bin/env bash
# Ablation: remove one module at a time
set -e
COMMON="--trainset DIV2K --testset kodak --train-data-dir data/DIV2K_train_HR --test-data-dir data/kodak \
--C 96 --multiple-snr 0,5,10,15,20 --snr-sample-mode fading_multi --fading-blocks 8"

# w/o SQA
python main.py --training \ --model SwinJSCC_w/o_SAandRA \
  --sqa-mode none --sqa-scope none --sfrc-mode sfrc --sfrc-scope channel_side2

# w/o SFRC
python main.py --training \ --model SwinJSCC_w/o_SAandRA \
  --sqa-mode emb_caq --sqa-scope channel_side2 --sfrc-mode none --sfrc-scope none

# backbone only
python main.py --training \ --model SwinJSCC_w/o_SAandRA \
  --sqa-mode none --sqa-scope none --sfrc-mode none --sfrc-scope none