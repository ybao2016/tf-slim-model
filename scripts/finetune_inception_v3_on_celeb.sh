#!/bin/bash
#
# This script performs the following operations:
# 1. Downloads the Flowers dataset
# 2. Fine-tunes an InceptionV3 model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.
#
# Usage:
# cd slim
# ./scripts/finetune_inceptionv3_on_celeb.sh

# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/home/nick/datasets/pretrained_data/checkpoints

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/old_home/nick/tmp/celeb-models/inception_v3

# Where the dataset is saved to.
DATASET_DIR=/home/nick/datasets/celeb

# Download the pre-trained checkpoint.
#if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
#  mkdir ${PRETRAINED_CHECKPOINT_DIR}
#fi
#if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt ]; then
#  wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
#  tar -xvf inception_v3_2016_08_28.tar.gz
#  mv inception_v3.ckpt ${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt
#  rm inception_v3_2016_08_28.tar.gz
#fi

# Download the dataset
#python download_and_convert_data.py \
#  --dataset_name=celeb \
#  --dataset_dir=${DATASET_DIR}

t1_last_layer_start="$(date +%s)"

# Fine-tune only the new layers for ${LAST_LAYER_STEPS} steps.
LAST_LAYER_STEPS=5000

python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=celeb \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/model.ckpt-10000 \
  --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --max_number_of_steps=${LAST_LAYER_STEPS} \
  --batch_size=32 \
  --learning_rate=0.01 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=600 \
  --save_summaries_secs=1600 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

t2_last_layer_end="$(date +%s)"

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=celeb \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3

t3_first_eval_end="$(date +%s)"

# Fine-tune all the new layers for ${ALL_LAYER_STEPS} steps.
ALL_LAYER_STEPS=6380
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_name=celeb \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3 \
  --checkpoint_path=${TRAIN_DIR} \
  --max_number_of_steps=${ALL_LAYER_STEPS} \
  --batch_size=64 \
  --learning_rate=0.0001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=600 \
  --save_summaries_secs=1600 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

t4_all_layer_end="$(date +%s)"

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=celeb \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3

t5_second_eval_end="$(date +%s)"


# summarize the time used for each session
# train last layer
tt=$(($t2_last_layer_end - $t1_last_layer_start))
tt_h=$(($tt / 3600))
tt_m=$((($tt % 3600)/60))
tt_s=$(($tt % 60))
echo "train last layer time is:             $tt_h hours, $tt_m minutes, $tt_s seconds "

# first eval time
tt=$(($t3_first_eval_end - $t2_last_layer_end))
tt_h=$(($tt / 3600))
tt_m=$((($tt % 3600)/60))
tt_s=$(($tt % 60))
echo "first eval time is:                   $tt_h hours, $tt_m minutes, $tt_s seconds "

# train all layers
tt=$(($t4_all_layer_end - $t3_first_eval_end))
tt_h=$(($tt / 3600))
tt_m=$((($tt % 3600)/60))
tt_s=$(($tt % 60))
echo "train all layesr time is:             $tt_h hours, $tt_m minutes, $tt_s seconds "

# second eval time
tt=$(($t5_second_eval_end - $t4_all_layer_end))
tt_h=$(($tt / 3600))
tt_m=$((($tt % 3600)/60))
tt_s=$(($tt % 60))
echo "second eval time is:                  $tt_h hours, $tt_m minutes, $tt_s seconds "
