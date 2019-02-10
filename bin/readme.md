ssh hdpsbp@ai-prd-01 /users/hdpsbp/bk/df_jf/bin/check.sh 20  & 
ssh hdpsbp@ai-prd-02 /users/hdpsbp/bk/df_jf/bin/check.sh 20  & 
ssh hdpsbp@ai-prd-03 /users/hdpsbp/bk/df_jf/bin/check.sh 20  & 
ssh hdpsbp@ai-prd-04 /users/hdpsbp/bk/df_jf/bin/check.sh 20  & 
ssh hdpsbp@ai-prd-05 /users/hdpsbp/bk/df_jf/bin/check.sh 20  & 
ssh hdpsbp@ai-prd-06 /users/hdpsbp/bk/df_jf/bin/check.sh 20  & 
ssh hdpsbp@ai-prd-07 /users/hdpsbp/bk/df_jf/bin/check.sh 20  & 


python ./core/check.py -L --wtid 12 --mini 0 --thread 1

nohup python ./core/check.py -L --bin_count 5 --bin_id 1 --gp_name lr_bin5 > 8.log 2>&1 &