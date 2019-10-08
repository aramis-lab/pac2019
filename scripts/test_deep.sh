#!/usr/bin/env bash
# Script used to estimate the ages of unlabeled data

TSVPATH="/path/to/tsv"
RESULTSPATH="/path/to/results_dir"
IMGPATH="/path/to/images" # A list of image directories may be given

python /path/to/git_clone/src/deep/test_evaluation.py $TSVPATH $IMGPATH RESULTSPATH
                                                      # --split_list 0 \ # Uncomment this line to give a list of folds

                                                      --gpu \ # Comment this line to use CPU instead of GPU
                                                      --num_threads 0 \
                                                      --num_workers 1

