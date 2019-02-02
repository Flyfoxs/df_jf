cd "$(dirname "$0")"

cd ..

python -u ./core/train.py --version 0129_v4 $* >> train_"$(hostname)".log 2>&1

# nohup ./bin/train.sh --file_num 0 --cut_len 100 --top_threshold 0.6  &


#python -u ./core/train.py \
#  --file_num 1 \
#  --cut_len 100 \
#  --top_threshold 0.6 \
#  >> train_"$(hostname)".log 2>&1
#
#
#python -u ./core/train.py \
#  --file_num 0 \
#  --cut_len 200 \
#  --top_threshold 0.6 \
#  >> train_"$(hostname)".log 2>&1
