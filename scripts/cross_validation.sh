MERGED="/path/to/pac_csv"
DATAPATH="/path/to/tsv_output"

PVAL=0.85

python ~/Code/pac2019/src/cross_validation/kfold_split.py $MERGED $DATAPATH -p $PVAL
