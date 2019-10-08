#!/usr/bin/env bash
SCRIPT="train.py"
MODEL="ResNet"

TSVPATH="/path/to/tsv"
RESULTSPATH="/path/to/results_dir"
IMGPATH="/path/to/images" # A list of image directories may be given

python /path/to/git_clone/src/deep/$SCRIPT $TSVPATH $RESULTSPATH $IMGPATH $MODEL\

                                           --sampler "random" \
                                           --data_normalization MinMax \ # Remove this line not to normalize data
                                           --output_normalization ""
                                           --blacklist '/path/to/blacklist_file' # Remove this line to use all subjects
                                           --file_extension 'file_ext' # Specify file extension only if the images are
                                           # not of format <subject_name> + "_" + <img_dir> + ".pt". In this case will
                                           # replace <img_dir> by any string specified.
                                           # --selection "old-40" # Uncomment this line to select subjects used for
                                           # training based on their age.

                                           --n_splits 5 \
                                           # --split 0 \ # Uncomment this line to specify the list of folds to compute

                                           --epochs 20 \
                                           --patience 10 \
                                           --tolerance 0.05 \
                                           --dropout 0 \

                                           --optimizer "Adam"
                                           --learning_rate 0.01 \
                                           --weight_decay 1e-4 \
                                           --loss "L1Loss" \
                                           -asteps 1 \

                                           -esteps 100 \

                                           --gpu \ # Comment this line to use CPU instead of GPU
                                           --batch_size 4 \
                                           --num_threads 0 \
                                           --num_workers 1 \
                                           --training_evaluation "whole_set"
