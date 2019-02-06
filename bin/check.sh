cd "$(dirname "$0")"

cd ..

PYTHONPATH=/users/hdpsbp/bk/df_jf:/users/hdpsbp/felix/keras:$PYTHONPATH

PATH=/apps/dslab/anaconda/python3/bin:$PATH
python ./core/check.py -L --wtid $1  $2 $3 $4 $5 $6 > log/score_"$(hostname)".log 2>&1
