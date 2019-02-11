#!/bin/bash

PYTHONPATH=/users/hdpsbp/bk/df_jf:/users/hdpsbp/felix/keras:$PYTHONPATH

PATH=/apps/dslab/anaconda/python3/bin:$PATH

bin_count=$1

for i in $( seq 0 "$(($bin_count-1))" )
do
    echo $i
    python ./core/check.py -L                   \
                        --bin_count $bin_count  \
                        --bin_id $i             \
                        --gp_name lr_bin_$bin_count     \
                        > log/bin_"$(hostname)"_$i.log 2>&1
done

# nohup ./bin/check_mini.sh 5 &
