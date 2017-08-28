#!/bin/bash
#
# 3. Evaluates the model on the Nickf validation set.
#
# Usage:
# cd slim
# ./scripts/eval_inceptionv3_on_nickf.sh

# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/home/nick/datasets/pretrained_data/checkpoints

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/old_home/nick/tmp/nick_family-models/inception_v3

# Where the dataset is saved to.
DATASET_DIR=/home/nick/datasets/nickf

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=nickf \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3
