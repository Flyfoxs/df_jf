#!/bin/bash
cd "$(dirname "$0")"

cd ..
PYTHONPATH=/users/hdpsbp/HadoopDir/felix/df_jf:/users/hdpsbp/felix/keras:$PYTHONPATH

PATH=/apps/dslab/anaconda/python3/bin:$PATH



for bin_count in {0..9}
do
    echo $bin_count

    python ./core/predict.py --class_name lr $1 $2 $3  >> ./log/ind_"$(hostname)"_$bin_count.log 2>&1
done

