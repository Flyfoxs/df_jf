cd "$(dirname "$0")"

cd ..

python ./core/validate.py --check_cnt 1 > val.log 2>&1
python ./core/validate.py --check_cnt 1 >> val.log 2>&1


python ./core/validate.py --check_cnt 2 >> val.log 2>&1
python ./core/validate.py --check_cnt 2 >> val.log 2>&1
