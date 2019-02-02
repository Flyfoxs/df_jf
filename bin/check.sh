cd "$(dirname "$0")"

cd ..

PYTHONPATH=/users/hdpsbp/bk/df_jf:/users/hdpsbp/felix/keras:$PYTHONPATH


/apps/dslab/anaconda/python3/bin/python ./core/check.py -L --col_begin $1 --col_end $2
