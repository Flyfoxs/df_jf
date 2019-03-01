cd "$(dirname "$0")"

cd ..


python ./core/validate.py --shift 3   --gp_name lr_bin_9 >   val_3.log 2>&1
python ./core/validate.py --shift 2   --gp_name lr_bin_9  >   val_2.log 2>&1
python ./core/validate.py --shift 1 --gp_name lr_bin_9 > val_1.log 2>&1


