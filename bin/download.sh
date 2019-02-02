cd "$(dirname "$0")"

cd ..


#rsync -av  hdpsbp@ai-prd-07:/users/hdpsbp/bk/df_jf/cache ./

rsync -av  hdpsbp@ai-prd-07:/users/hdpsbp/bk/df_jf/output/*.zip ./output
