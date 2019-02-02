cd "$(dirname "$0")"

cd ..

PYTHONPATH=/users/hdpsbp/bk/df_jf:/users/hdpsbp/felix/keras:$PYTHONPATH

PATH=/apps/dslab/anaconda/python3/bin:$PATH
python ./core/check.py -L --col_begin $1 --col_end $2 $3 $4 $5 $6
