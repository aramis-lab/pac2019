#!/usr/bin/env bash
# Script used to run the evaluation of data used for the k-fold

RESULTSPATH="/path/to/results_dir"

python /path/to/git_clone/src/deep/evaluation.py $RESULTSPATH

                                                 --train_mode \ # Comment this line to use validation data
                                                 # --split_list 0 \ # Uncomment this line to specify a list of folds

                                                 --gpu \ # Comment this line to use CPU instead of GPU
                                                 --num_threads 0 \
                                                 --num_workers 1
