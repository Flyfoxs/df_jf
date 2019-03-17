cd "$(dirname "$0")"

cd ..

date
#rsync -av  hdpsbp@ai-prd-07:/users/hdpsbp/bk/df_jf/cache ./

rsync -av  hdpsbp@ai-prd-05:/users/hdpsbp/felix/df_jf/cache/*template*  ./cache/

rsync -av  hdpsbp@ai-prd-05:/users/hdpsbp/felix/df_jf/cache/*block*  ./cache/

#rsync -av  hdpsbp@ai-prd-05:/users/hdpsbp/felix/df_jf/imp/*  ./imp/
#rsync -av  hdpsbp@ai-prd-05:/users/hdpsbp/felix/df_jf/output/*.zip ./output
#
#rsync -av  hdpsbp@ai-prd-05:/users/hdpsbp/felix/df_jf/output/good_luck.csv_* ./output


#rsync -av  hdpsbp@ai-prd-05:/users/hdpsbp/felix/df_jf/output/*0218*.csv ./output

#nohup rsync -av  hdpsbp@ai-prd-07:/users/hdpsbp/bk/df_jf/score/lr*9 ./score/ &
date