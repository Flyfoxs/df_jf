#!/bin/bash

PYTHONPATH=/users/hdpsbp/HadoopDir/felix/df_jf:/users/hdpsbp/felix/keras:$PYTHONPATH

PATH=/apps/dslab/anaconda/python3/bin:$PATH



for bin_count in {0..9}
do
    echo $bin_count
    cd /users/hdpsbp/HadoopDir/felix/df_jf
    python ./core/predict.py --class_name lr >> ~/log/ind_"$(hostname)"_$bin_count.log 2>&1
done

